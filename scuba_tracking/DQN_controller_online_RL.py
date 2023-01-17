#!/usr/bin/env python
############################
# online RL controller; this code includes exploration and exploitation to gether with a PID controller to
# keep the system in an acceptable region of performance
############################
import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node

############## RL part ########################
from collections import namedtuple, deque
from itertools import count
import random
import math
import torch
import torch.nn as nn
from network import DQN_JOINT
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch.optim as optim
##################################################
# if source install/setup.bash the msg_ws -> then
from ir_msgs.msg import Command
from std_msgs.msg import String

import os, time
import numpy as np
####################################### export PYTHONPATH=/home/USERNAME/sim_ws
from src.scuba_tracking.scuba_tracking.utils.controller_utils import PID_controller, msg_processing

Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'yaw_reward', 'pitch_reward'))

class controller(Node):

    def __init__(self):
        super().__init__('controller')

        self.direct_command = Command()
        self.direct_command.speed = 0.0
        self.direct_command.roll = 0.0
        self.direct_command.yaw = 0.0
        self.direct_command.heave = 0.0
        self.direct_command.pitch = 0.0

        ### To gather data while exploring the space by our controller
        self.path_to_gathered_data = './sampled_scenarios/'
        if not os.path.isdir(self.path_to_gathered_data):
            os.mkdir(self.path_to_gathered_data)
            self.num_of_experiments = 0
        else:
            self.num_of_experiments = len(os.listdir(self.path_to_gathered_data))
        #workbook = xlsxwriter.Workbook('scuba_tracking#1.xlsx')
        #self.excel_writer = workbook.add_worksheet()
        self.sample_counter = 0

        self.debug = True
        self.image_size = (400,300)
        self.single_object_tracking = True
        # let's say it's the indirect distance to the diver
        self.BB_THRESH = 4800

        self.vision_output_subscription = self.create_subscription(
            String,
            '/aqua/detected_objects',
            self.data_handler,
            30)

        self.pose_subscription = self.create_subscription(
            Vector3,
            '/simulator/position_ground_truth',
            self.pose_callback,
            30)

        self.command_publisher = self.create_publisher(Command, '/aqua/command', 30)
        # This current_state_publisher is to make sure that we have the pair of (state,action)
        self.current_state_publisher = self.create_publisher(String, '/aqua/current_state', 30)
        # CONTROLLER PART
        self.controller = PID_controller()
        # RL Controller
        self.RL_controller = DQN_approach()
        self.obs = np.zeros(self.history_buffer_size * self.num_of_states)
        self.previous_action = []
        self.previous_state = []
        self.actions_list_yaw = np.linspace(-0.8, 0.8, self.num_of_actions, True)
        self.actions_list_pitch = np.linspace(-0.4, 0.4, self.num_of_actions, True)
        self.reset_recovery_variables()

    def reset_recovery_variables(self):
        # For the Recovery part
        self.lost_target_step = 0
        self.rightSideChecked = False
        self.begin_time = time.time()
        self.changing_time = 2  # 2 seconds for the beginning

    def reward_calculation(self, current_observation):
        try:
            factor_ = 0.1
            # yaw part
            if abs(current_observation[0]) < 0.01:
                reward = 1
            else:
                # it's important to keep our rewards smaller than one to have converged Q values
                # positive reward:
                reward = factor_*(1 / (abs(current_observation[0]) + factor_))
                # negative reward:
                #reward = -0.49*(abs(current_observation[0]) + abs(current_observation[1]))
            yaw_reward = reward

            # pitch part
            if abs(current_observation[1]) < 0.01:
                reward = 1
            else:
                # it's important to keep our rewards smaller than one to have converged Q values
                # positive reward:
                reward = factor_*(1 / (abs(current_observation[1]) + factor_))
                # negative reward:
                #reward = -0.49*(abs(current_observation[0]) + abs(current_observation[1]))
            pitch_reward = reward
        except:
            print('object lost! reward = -1')
            yaw_reward = -100
            pitch_reward = -100

        return yaw_reward, pitch_reward


    def get_action(self, yaw_rate, pitch_rate,discrete=True):
        if discrete:
            yaw_rate = np.argmin(abs(yaw_rate - self.RL_actions_list_yaw))
            pitch_rate = np.argmin(abs(pitch_rate - self.RL_actions_list_pitch))
            return yaw_rate, pitch_rate
        else:
            return yaw_rate, pitch_rate

    def data_handler(self, msg):
        yaw_ref = 0.0
        pitch_ref = 0.0
        speed_ref = 0.0
        # just to know how msg.data looks like:
        #1#102.01816,197.34833,214.18144,264.59863#
        #num_of_objs#obj1_bb#obj2_bb#...#
        mean_of_obj_locations = msg_processing(msg)
        if mean_of_obj_locations[0]>-1:
            if self.single_object_tracking:

                yaw_ref, pitch_ref, speed_ref = self.controller(mean_of_obj_locations)
                ################## For the recovery part #######################
                self.reset_recovery_variables()
            else:
                pass #TODO -> TWO scuba divers at the same time
        else:
            # The spiral search strategy
            yaw_ref = self.search()
            self.lost_target_step += 1

        ######################### preparing the input for RL ################################
        # Add some noises to improve the NN input
        bb_noise_x = np.random.randint(0, 5)
        bb_noise_y = np.random.randint(0, 5)
        area_noise = bb_noise_x * bb_noise_y  # this I think must be dependent on the two previous ones
        speed_noise = 0.02 * np.random.rand(3)

        current_observation = np.array([mean_of_obj_locations[0] + bb_noise_x,
                                        mean_of_obj_locations[1] + bb_noise_y,
                                        mean_of_obj_locations[2] + area_noise,
                                        speed_ref + speed_noise])

        # now normalization
        # TODO no normalization on the  velocities :| lin_vel > 1 ???!
        # 2* is to end up with a value in [-1 , 1]
        current_observation[0] = 2*(current_observation[0] - self.image_size[0] / 2) / self.image_size[0]
        current_observation[1] = 2*(current_observation[1] - self.image_size[1] / 2) / self.image_size[1]
        current_observation[2] = 2*(current_observation[2] - self.image_area / 2) / self.image_area

        self.obs[:-self.num_of_states] = self.obs[self.num_of_states:]
        self.obs[-self.num_of_states:] = current_observation

        state = torch.tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        ############# check the situation to be controllable by RL at low risk of losing the target #############
        if ((0.15 * self.image_size[0] < mean_of_obj_locations[0] < 0.85 * self.image_size[0]) and
                (0.15 * self.image_size[1] < mean_of_obj_locations[1] < 0.85 * self.image_size[1])):
            NN_output = self.RL_controller.select_action(state)
            yaw_ref = self.RL_actions_list_yaw[NN_output[0].view(-1)[0].cpu().detach().numpy()]  # this we don't use now
            pitch_ref = self.RL_actions_list_pitch[NN_output[1].view(-1)[0].cpu().detach().numpy()]  # this we don't use now

        ########################## apply the control to the robot #############################
        self.direct_command.yaw = yaw_ref
        self.direct_command.pitch = pitch_ref
        self.direct_command.speed = speed_ref
        self.direct_command.roll = 0.0
        if self.debug:
            print('speed ref: ', speed_ref)

        self.command_publisher.publish(self.direct_command)
        self.current_state_publisher.publish(msg)

        ####################### online RL part ########################################################
        # if this is the first time then just save the first state
        if not (len(self.previous_state) == 0):
            ## reward calculation
            yaw_reward, pitch_reward = self.reward_calculation(mean_of_obj_locations)
            # Store the transition in memory self.previous_state, self.previous_action, state, reward
            self.RL_controller.learn(self.previous_state, self.previous_action, state, yaw_reward, pitch_reward)

        self.previous_state = state #TODO check this!!!! to be updated
        self.previous_action = torch.tensor([self.get_action(yaw_ref, pitch_ref)], device=self.device, dtype=torch.long)


    # for log purposes
    def pose_callback(self, msg):
        x, y, z = msg.x, msg.y, msg.z

        #print(x, y, z)
    def search(self): # return yaw rate - SPIRAL SEARCH
        # Change the view direction every so often (self.changing_time)
        if time.time() - self.begin_time>self.changing_time:
            self.begin_time = time.time()
            # Note, *2 is not enough as you have to also take into account the arrival time to the origin
            if self.lost_target_step:
                self.changing_time = self.changing_time + 2*self.changing_time
            else:
                self.changing_time *= 2
            self.rightSideChecked = not self.rightSideChecked

        if self.rightSideChecked:
            # right direction
            return 0.5
        else:
            # left direction
            return -0.5

class DQN_approach:
    def __init__(self):

        self.reset()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_hyperparameters()
        self.num_of_actions = 9
        self.history_buffer_size = 3
        self.num_of_states = 4  # center_x, center_y, area_of_diver_bb, linear_vel
        self.num_of_actions = 9
        self.obs_dim = self.num_of_states * self.history_buffer_size
        self.act_dim = self.num_of_actions

        self.policy_net = DQN_JOINT(self.obs_dim, [self.act_dim, self.act_dim]).to(self.device)
        self.target_net = DQN_JOINT(self.obs_dim, [self.act_dim, self.act_dim]).to(self.device)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        self.debug_mode = False
        if self.debug_mode:
            summary(self.policy_net, (1, self.obs_dim))

        self.CHECKPOINT_PATH = './trained_models/training_checkpoint'

        try:
            checkpoint = torch.load(self.CHECKPOINT_PATH, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['model_state_dict_policy'], strict=True)
            self.target_net.load_state_dict(checkpoint['model_state_dict_target'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.starting_episode = checkpoint['TRAINING_STEP']
            print('Weights loaded!')
        except:
            print('No checkpoint found!')
            self.starting_episode = 0

        self.min_ERM_size = 10000
        self.writer_train = SummaryWriter('./runs/training')

    def _init_hyperparameters(self):
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the AdamW optimizer
        self.BATCH_SIZE = 50
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.ACTION_SCALER = 2
        self.num_episodes = 50000

    def reset(self):
        self.ERM = ReplayMemory(30000)
        self.steps_done = 0

    def select_action(self,state):

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                network_output = self.policy_net(state)
                return network_output[0].max(1)[1].view(1, 1), network_output[1].max(1)[1].view(1, 1)
        else:
            # now we have two actions
            return [torch.tensor([[int(np.random.randint(0,self.act_dim))]], device=self.device, dtype=torch.long), \
                torch.tensor([[int(np.random.randint(0, self.act_dim))]], device=self.device, dtype=torch.long)]


    def optimize_model(self):
        if len(self.ERM) < (self.min_ERM_size):
            return

        transitions = self.ERM.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        yaw_reward_batch = torch.cat(batch.yaw_reward)
        pitch_reward_batch = torch.cat(batch.pitch_reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        # A BIG ERROR! :||||||||||||| .gather uses the action_batch to select the corresponding output from a list meaning that
        # a negative control output may destroy everything
        #####state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        ######################################## WE WILL HAVE TWO SETS OF ACTION VALUES ##############################
        state_action_values = self.policy_net(state_batch)
        state_action_values_yaw = state_action_values[0].gather(1, action_batch[:,0].unsqueeze(-1))
        state_action_values_pitch = state_action_values[1].gather(1, action_batch[:,1].unsqueeze(-1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values_yaw = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values_pitch = torch.zeros(self.BATCH_SIZE, device=self.device)

        with torch.no_grad():
            outputs = self.target_net(non_final_next_states)
            next_state_values_yaw[non_final_mask] = outputs[0].max(1)[0]
            next_state_values_pitch[non_final_mask] = outputs[1].max(1)[0]
        # Compute the expected Q values
        expected_state_action_values_yaw = (next_state_values_yaw * self.GAMMA) + yaw_reward_batch
        expected_state_action_values_pitch = (next_state_values_pitch * self.GAMMA) + pitch_reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        yaw_loss = criterion(state_action_values_yaw, expected_state_action_values_yaw.unsqueeze(1))
        pitch_loss = criterion(state_action_values_pitch, expected_state_action_values_pitch.unsqueeze(1))

        loss = yaw_loss + pitch_loss
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.steps_done += 1

        return loss

    def learn(self, previous_state, previous_action, state, yaw_reward, pitch_reward):

        # Store the transition in memory
        self.ERM.push(previous_state, previous_action, state, yaw_reward, pitch_reward)

        #################################################################
        # Perform one step of the optimization (on the policy network)
        loss = self.optimize_model()
        #################################################################
        if not loss is None:
            self.writer_train.add_scalar('Total Loss', loss, self.steps_done)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                        1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        if self.steps_done%10:
            torch.save({
                'TRAINING_STEP': self.steps_done,
                'model_state_dict_policy': self.policy_net.state_dict(),
                'model_state_dict_target': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(), }, self.CHECKPOINT_PATH)



class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def main():

    rclpy.init()

    node_ = controller()

    rclpy.spin(node_)

    node_.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
