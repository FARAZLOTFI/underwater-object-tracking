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

from src.scuba_tracking.scuba_tracking.RL_network import DQN_JOINT
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

from config import config

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
        self.trajectory = []
        #workbook = xlsxwriter.Workbook('scuba_tracking#1.xlsx')
        #self.excel_writer = workbook.add_worksheet()
        self.sample_counter = 0

        self.debug = True
        self.image_size = config.IMAGE_SIZE
        self.image_area = self.image_size[0]*self.image_size[1]
        self.single_object_tracking = True
        # let's say it's the indirect distance to the diver
        self.BB_THRESH = config.BB_AREA_THRESHOLD
        self.target_x = None
        self.target_y = None
        self.target_area = None

        self.vision_output_subscription = self.create_subscription(
            String,
            config.GENERATED_BB_TOPIC,
            self.data_handler,
            30)

        self.pose_subscription = self.create_subscription(
            Vector3,
            config.ROBOT_POS_TOPIC,
            self.pose_callback,
            30)

        self.command_publisher = self.create_publisher(Command, config.COMMAND_TOPIC, 10)
        # This current_state_publisher is to make sure that we have the pair of (state,action)
        self.current_state_publisher = self.create_publisher(String, '/aqua/current_state', 10)
        # CONTROLLER PART
        self.controller = PID_controller()
        # RL Controller
        self.RL_controller = DQN_approach()
        self.obs = np.zeros(self.RL_controller.history_buffer_size * self.RL_controller.num_of_states)
        self.previous_action = []
        self.previous_state = []

        self.RL_actions_list_yaw = np.linspace(config.MIN_YAW_RATE, config.MAX_YAW_RATE, self.RL_controller.num_of_actions, True)
        self.RL_actions_list_pitch = np.linspace(config.MIN_PITCH_RATE, config.MAX_PITCH_RATE, self.RL_controller.num_of_actions, True)
        self.reset_recovery_variables()

    # We take the last object location to have an estimation of where it should be if was lost
    def reset_recovery_variables(self, last_obj_location=None):
        # For the Recovery part
        self.lost_target_step = 0
        self.begin_time_right = time.time()
        self.begin_time_up = time.time()

        self.changing_time_right = 3  # 2 seconds for the beginning
        self.changing_time_up = 3  # 2 seconds for the beginning

        if last_obj_location is not None:
            # we add a 10 pixel threshold to increase the certainty
            if last_obj_location[0]>(self.image_size[0]/2 + 10):
                self.rightSideChecked = False
            else:
                self.rightSideChecked = True
            # instead of 10 we use 5 pixel threshold as height is smaller than width
            if last_obj_location[1]>(self.image_size[1]/2 + 5):
                self.upSideChecked = True
            else:
                self.upSideChecked = False

        else:
            self.rightSideChecked = False
            self.upSideChecked = False

    def reward_calculation(self, current_observation):
        try:
            factor_ = 0.1
            # yaw part
            if abs(current_observation[0]) < 0.05:
                reward = 1
            else:
                # it's important to keep our rewards smaller than one to have converged Q values
                # positive reward:
                reward = factor_*(1 / (abs(current_observation[0]) + factor_))
                # negative reward:
                ##reward = -0.5*(abs(current_observation[0]))
            yaw_reward = reward

            # for out of plane cases
            if abs(current_observation[0])>0.95:
                yaw_reward = -0.1
            #####################################################################
            # pitch part
            if abs(current_observation[1]) < 0.05:
                reward = 1
            else:
                # it's important to keep our rewards smaller than one to have converged Q values
                # positive reward:
                reward = factor_*(1 / (abs(current_observation[1]) + factor_))
                # negative reward:
                ##reward = -0.5*(abs(current_observation[1]))
            pitch_reward = reward
            if abs(current_observation[1])>0.95:
                pitch_reward = -0.1
        except:
            print('object lost! reward = -1')
            yaw_reward = -1
            pitch_reward = -1

        #general_reward = 1/(2*abs(current_observation[1])**2 +  2*abs(current_observation[0])**2 + 1)
        #general_reward, general_reward
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
        # the default mode is to keep the target at the center point
        if not config.PID_RANDOM_TARGET_MODE:
            self.target_x = None
            self.target_y = None
            self.target_area = None
        else:
            # Otherwise, for exploration purposes
            if self.sample_counter%200 == 0 :
                self.target_x = np.random.randint(0.2*self.image_size[0], 0.8*self.image_size[0])
                self.target_y = np.random.randint(0.2*self.image_size[1], 0.8*self.image_size[1])
                self.target_area = np.random.randint(0.7*self.BB_THRESH, 1.2*self.BB_THRESH)
                print('Target updated! ',[self.target_x,self.target_y,self.target_area])

        spiral_search_flag = False
        if mean_of_obj_locations[0]>-1:
            if self.single_object_tracking:
                yaw_ref, pitch_ref, speed_ref = self.controller(mean_of_obj_locations, self.target_x, self.target_y, self.target_area)
                # to have saturated PID controllers as adversarial agents
                if config.SATURATED_PID:
                    if yaw_ref<0:
                        yaw_ref = np.maximum(config.MIN_YAW_RATE,yaw_ref)
                    else:
                        yaw_ref = np.minimum(config.MAX_YAW_RATE,yaw_ref)

                    if pitch_ref<0:
                        pitch_ref = np.maximum(config.MIN_PITCH_RATE,pitch_ref)
                    else:
                        pitch_ref = np.minimum(config.MAX_PITCH_RATE,pitch_ref)

                print('PID: ', [yaw_ref, pitch_ref])
                ################## For the recovery part #######################
                self.reset_recovery_variables(mean_of_obj_locations)
            else:
                pass #TODO -> TWO scuba divers at the same time
        else:
            # The spiral search strategy
            yaw_ref, pitch_ref, speed_ref = self.search()
            print('spiral search: ',[yaw_ref,pitch_ref])
            # in this short part for cases of lost target we use an artifical input for the RL controller
            # this is to make the neural network aware of the target being lost
            if self.rightSideChecked: # it has been lost on the left side
                right_seen = 0.0
            else:
                right_seen = self.image_size[0]

            if self.upSideChecked:  # it has been lost on the left side
                up_seen = self.image_size[1]
            else:
                up_seen = 0.0

            #-25 is set based on the noises considered in the following
            mean_of_obj_locations = np.array([right_seen, up_seen, -25.0])
            self.lost_target_step += 1

            spiral_search_flag = True
        ######################### preparing the input for RL ################################
        # Add some noises to improve the NN input
        bb_noise_x = np.random.randint(0, 5)
        bb_noise_y = np.random.randint(0, 5)
        area_noise = bb_noise_x * bb_noise_y  # this I think must be dependent on the two previous ones
        speed_noise = 0.01 * np.random.rand() # 3

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
        self.obs[:-self.RL_controller.num_of_states] = self.obs[self.RL_controller.num_of_states:]
        self.obs[-self.RL_controller.num_of_states:] = current_observation

        state = torch.tensor(self.obs, dtype=torch.float32, device=self.RL_controller.device).unsqueeze(0)
        
        ############# check the situation to be controllable by RL at low risk of losing the target #############
        PID_con_contribution = 0.0# max -> 0.5
        PID_random_contribution = 0.4 # max -> 1
        if ((PID_con_contribution * self.image_size[0] < mean_of_obj_locations[0] < (1-PID_con_contribution) * self.image_size[0]) and
                (PID_con_contribution * self.image_size[1] < mean_of_obj_locations[1] < (1-PID_con_contribution) * self.image_size[1])) \
                and np.random.rand()>PID_random_contribution:
            NN_output = self.RL_controller.select_action(state)
            yaw_ref = self.RL_actions_list_yaw[NN_output[0].view(-1)[0].cpu().detach().numpy()]  # this we don't use now
            pitch_ref = self.RL_actions_list_pitch[NN_output[1].view(-1)[0].cpu().detach().numpy()]  # this we don't use now
            print('RL: ',[yaw_ref,pitch_ref])
        ########################## apply the control to the robot #############################
        # saturated input control to limit the power of the PID controller
        self.direct_command.yaw = yaw_ref #>0 right
        self.direct_command.pitch = pitch_ref #>0 down
        self.direct_command.speed = speed_ref
        self.direct_command.roll = 0.1
        if self.debug:
            print('speed ref: ', speed_ref)
            pass

        self.command_publisher.publish(self.direct_command)
        self.current_state_publisher.publish(msg)

        ####################### online RL part ########################################################
        # if this is the first time then just save the first state
        if not (len(self.previous_state) == 0) and not(spiral_search_flag):
            ## reward calculation
            yaw_reward, pitch_reward = self.reward_calculation(current_observation)
            print('rewards: ',yaw_reward,pitch_reward)
            # Store the transition in memory self.previous_state, self.previous_action, state, reward


            self.RL_controller.ERM.push(self.previous_state, self.previous_action, self.obs, yaw_reward, pitch_reward)
            if self.sample_counter % 1000 == 0:
                #np.save(self.RL_controller.path_to_gathered_data + str(self.RL_controller.num_of_experiments), self.RL_controller.ERM.memory)
                print('ERM saved!')
                #np.save(self.RL_controller.path_to_gathered_data + 'scenario#' +str(self.RL_controller.num_of_experiments), self.trajectory)
            self.RL_controller.learn()

            # the sample counter is used to update a random target for the PID controllers
            self.sample_counter += 1

        self.previous_state = np.zeros(len(self.obs))
        self.previous_state = self.previous_state + self.obs #TODO check this!!!! to be updated
        self.previous_action = self.get_action(yaw_ref, pitch_ref)
        #print('locations: ',mean_of_obj_locations)

    # for log purposes
    def pose_callback(self, msg):
        x, y, z = msg.x, msg.y, msg.z
        self.trajectory.append([x,y,z])
        #print(x, y, z)
    def search(self): # return yaw rate - SPIRAL SEARCH
        # Change the view direction every so often (self.changing_time)
        if time.time() - self.begin_time_right>self.changing_time_right:
            self.begin_time_right = time.time()
            # Note, *2 is not enough as you have to also take into account the arrival time to the origin
            if self.lost_target_step == 0:
                self.changing_time_right = self.changing_time_right + 2*self.changing_time_right
            else:
                self.changing_time_right *= 2
                self.rightSideChecked = not self.rightSideChecked

        if time.time() - self.begin_time_up>self.changing_time_up:
            self.begin_time_up = time.time()
            # Note, *2 is not enough as you have to also take into account the arrival time to the origin
            if self.lost_target_step == 0:
                self.changing_time_up = self.changing_time_up + 2*self.changing_time_up
            else:
                self.changing_time_up *= 2
                self.upSideChecked = not self.upSideChecked

        yaw_rate = 0.0
        pitch_rate = 0.0
        lin_vel = 0.0
        if self.rightSideChecked:
            # right direction
            yaw_rate = -1.0
        else:
            # left direction
            yaw_rate = 1.0

        if self.upSideChecked:
            # go to the down direction
            pitch_rate = 0.1
        else:
            pitch_rate = -0.1
        return yaw_rate, pitch_rate, lin_vel

class DQN_approach:
    def __init__(self):

        self.reset()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_hyperparameters()
        self.history_buffer_size = 5
        self.num_of_states = 4  # center_x, center_y, area_of_diver_bb, linear_vel
        self.num_of_actions = 5
        self.obs_dim = self.num_of_states * self.history_buffer_size
        self.act_dim = self.num_of_actions

        self.policy_net = DQN_JOINT(self.obs_dim, [self.act_dim, self.act_dim]).to(self.device)
        self.target_net = DQN_JOINT(self.obs_dim, [self.act_dim, self.act_dim]).to(self.device)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        self.debug_mode = False
        if self.debug_mode:
            summary(self.policy_net, (1, self.obs_dim))

        self.CHECKPOINT_PATH = config.RL_CHECKPOINT

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

        self.writer_train = SummaryWriter('./runs/training')

        ### To gather data while exploring the space by our controller
        self.path_to_gathered_data = './sampled_scenarios_from_RL_agent/'
        if not os.path.isdir(self.path_to_gathered_data):
            os.mkdir(self.path_to_gathered_data)
            self.num_of_experiments = 0
        else:
            self.num_of_experiments = len(os.listdir(self.path_to_gathered_data))

    def _init_hyperparameters(self):
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the AdamW optimizer
        self.BATCH_SIZE = 10
        self.GAMMA = 0.5
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.001
        self.LR = 1e-5

    def reset(self):
        self.ERM = ReplayMemory(500)
        self.steps_done = 0

    def select_action(self,state):

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if np.random.rand()>0.05:#sample > eps_threshold:
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
        if len(self.ERM) < self.BATCH_SIZE:
            return

        transitions = self.ERM.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        try:
            non_final_next_states = torch.tensor(batch.next_state,device=self.device, dtype=torch.float32)
            state_batch = torch.tensor(batch.state, device=self.device, dtype=torch.float32)
            action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.long)
            yaw_reward_batch = torch.tensor(batch.yaw_reward,device=self.device, dtype=torch.float32)
            pitch_reward_batch = torch.tensor(batch.pitch_reward, device=self.device, dtype=torch.float32)

        except:
            #print('Could not use this batch!!!! there has been a None value there')
            return
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

        with torch.no_grad():
            outputs = self.target_net(non_final_next_states)
            next_state_values_yaw = outputs[0].max(1)[0]
            next_state_values_pitch = outputs[1].max(1)[0]
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

    def learn(self):

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

        if self.steps_done%100:
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
