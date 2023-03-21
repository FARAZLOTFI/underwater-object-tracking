#!/usr/bin/env python
############################
# online RL controller; this code includes exploration and exploitation to gether with a PID controller to
# keep the system in an acceptable region of performance
############################
import rclpy
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseWithCovarianceStamped

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

class InvalidWindowSize(Exception):
    print('Window size must be greater than zero! ')
class reward_shaping:
    def __init__(self, window_size = 1, include_detection_confidence = 0.0 ):
        if window_size<1:
            print('episodic view')
        self.k_step = window_size
        self.include_detection_confidence = include_detection_confidence
        self.boundaries = config.IMAGE_ARTIFICAL_BOUNDARIES
        self.tracked_frame = 0
        self.yaw_reward = []
        self.pitch_reward = []
        self.flag_done = False
        self.sample_step = 0
    def __call__(self,current_observation, detection_conf = 0.0 ):
        yaw_reward, pitch_reward = self.single_observation(current_observation)

        if self.k_step > 1:

            if len(self.yaw_reward)>(self.k_step-1):
                self.yaw_reward[:-1] = self.yaw_reward[1:]
                self.yaw_reward[-1] = yaw_reward
                self.pitch_reward[:-1] = self.pitch_reward[1:]
                self.pitch_reward[-1] = pitch_reward

            else:
                self.yaw_reward.append(yaw_reward)
                self.pitch_reward.append(pitch_reward)

            if self.include_detection_confidence > 0:
                if detection_conf > 0.8:
                    reward_conf = 1.0
                elif detection_conf > 0.4:
                    reward_conf = detection_conf
                else:
                    reward_conf = 0.0
            else:
                reward_conf = 0.0
            scaler = 0
            #
            print('variance: ',np.array(self.yaw_reward).var(), np.array(self.pitch_reward).var(), reward_conf)
            return yaw_reward - scaler * np.array(self.yaw_reward).var() + self.include_detection_confidence * reward_conf , \
                   pitch_reward - scaler * np.array(self.pitch_reward).var() + self.include_detection_confidence * reward_conf , [yaw_reward, pitch_reward, detection_conf],self.flag_done

        else: # this part must be modified

            if abs(current_observation[0])<self.boundaries and abs(current_observation[1])<self.boundaries and (self.sample_step<config.MAX_EPISODE) :
                self.flag_done = False
                self.yaw_reward.append(yaw_reward)
                self.pitch_reward.append(pitch_reward)
                self.sample_step += 1 # samples per episode
                return yaw_reward, pitch_reward, [yaw_reward,pitch_reward],self.flag_done
            else:
                expected_yaw_rew = np.array(self.yaw_reward).sum()
                expected_pitch_rew = np.array(self.pitch_reward).sum()
                var_yaw_rew = np.array(self.yaw_reward).var()
                var_pitch_rew = np.array(self.pitch_reward).var()
                self.yaw_reward = []
                self.pitch_reward = []

                self.flag_done = True
                if self.sample_step < config.MAX_EPISODE:
                    yaw_reward = -1
                    pitch_reward = -1
                    if abs(current_observation[0])<self.boundaries:
                        pitch_reward = None
                    else:
                        pitch_reward = None

                self.sample_step = 0

                return yaw_reward, pitch_reward, [yaw_reward, pitch_reward],self.flag_done


    def single_observation(self, current_observation):
        try:
            factor_ = 0.1
            # yaw part
            if abs(current_observation[0]) < 0.05:
                reward = 1
            else:
                # it's important to keep our rewards smaller than one to have converged Q values
                # positive reward:
                reward = factor_ * (1 / (abs(current_observation[0]) + factor_))
                # negative reward:
                ##reward = -0.5*(abs(current_observation[0]))
            yaw_reward = reward

            # for out of plane cases
            if abs(current_observation[0]) > 0.9:
                yaw_reward = -0.1
            #####################################################################
            # pitch part
            if abs(current_observation[1]) < 0.05:
                reward = 1
            else:
                # it's important to keep our rewards smaller than one to have converged Q values
                # positive reward:
                reward = factor_ * (1 / (abs(current_observation[1]) + factor_))
                # negative reward:
                ##reward = -0.5*(abs(current_observation[1]))
            pitch_reward = reward
            if abs(current_observation[1]) > 0.9:
                pitch_reward = -0.1
        except:
            print('object lost! reward = -1')
            yaw_reward = -1
            pitch_reward = -1

        # general_reward = factor_*(1 / (abs(current_observation[0]) + factor_)) + factor_*(1 / (abs(current_observation[1]) + factor_))#1/(2*abs(current_observation[1])**2 +  2*abs(current_observation[0])**2 + 1)
        # general_reward, general_reward
        return yaw_reward, pitch_reward


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
            10)

        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            config.ROBOT_POS_TOPIC,
            self.pose_callback,
            10)

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
        # just for the sake of evaluation
        self.rewards_list = []

        self.reward_calculation = reward_shaping(0,0.0)
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
        mean_of_obj_locations, detection_confidence = msg_processing(msg)
        print('detection confidence:' ,detection_confidence)
        ######################################################
        SAFETY_MECHANISM = False
        ############### SAFETY MECHANISM ######################
        if (mean_of_obj_locations[2] - config.BB_AREA_MAX) > 0:
            SAFETY_MECHANISM = True
        # the default mode is to keep the target at the center point
        if not config.PID_RANDOM_TARGET_MODE:
            self.target_x = None
            self.target_y = None
            self.target_area = None
        else:
            # Otherwise, for exploration purposes
            if self.sample_counter%100 == 0 :
                self.target_x = np.random.randint(0.1*self.image_size[0], 0.9*self.image_size[0])
                self.target_y = np.random.randint(0.1*self.image_size[1], 0.9*self.image_size[1])
                self.target_area = np.random.randint(0.8*self.BB_THRESH, self.BB_THRESH)
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

            #-25 is set based oSAFETY_MECHANISMn the noises considered in the following
            mean_of_obj_locations = np.array([right_seen, up_seen, -25.0])
            self.lost_target_step += 1

            spiral_search_flag = True
        ######################### preparing the input for RL ################################
        # Add some noises to improve the NN input
        bb_noise_x = np.random.randint(0, 5)
        bb_noise_y = np.random.randint(0, 5)
        area_noise = bb_noise_x * bb_noise_y  # this I think must be dependent on the two previous ones
        speed_noise = 0.05 * np.random.rand() # 3
        #detection_confidence_noise = 0.1 * (np.random.rand() - 0.5)

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
        PID_con_contribution = config.IMAGE_ARTIFICAL_BOUNDARIES # max 1
        PID_random_contribution = 0.0 # max -> 1
        if (abs(current_observation[0])<PID_con_contribution and (abs(current_observation[1])<PID_con_contribution) \
                and np.random.rand()>PID_random_contribution):
            NN_output = self.RL_controller.select_action(state)
            yaw_ref = self.RL_actions_list_yaw[NN_output[0].view(-1)[0].cpu().detach().numpy()]  # this we don't use now
            pitch_ref = self.RL_actions_list_pitch[NN_output[1].view(-1)[0].cpu().detach().numpy()]  # this we don't use now
            print('RL: ',[yaw_ref,pitch_ref])
        ########################## apply the control to the robot #############################
        # saturated input control to limit the power of the PID controller
        self.direct_command.yaw = yaw_ref #>0 right
        self.direct_command.pitch = pitch_ref #>0 down
        self.direct_command.speed = speed_ref
        self.direct_command.roll = 0.0 * (np.random.rand() - 1)
        if self.debug:
            print('speed ref: ', speed_ref)
            pass

        ####################################### SAFETY MECHANISM ###########################
        if SAFETY_MECHANISM:
            self.direct_command.yaw = 0.0
            self.direct_command.pitch = 0.0
            self.direct_command.speed = 0.0
            self.direct_command.roll = 0.0
        ####################################################################################

        self.command_publisher.publish(self.direct_command)
        self.current_state_publisher.publish(msg)

        ####################### online RL part ########################################################
        # if this is the first time then just save the first state
        if not (len(self.previous_state) == 0) and not(spiral_search_flag) and not(SAFETY_MECHANISM):
            ## reward calculation

            yaw_reward, pitch_reward, for_evaluation, flag_done = self.reward_calculation(current_observation, detection_confidence)
            print('flag done: ',flag_done)
            print('detail: ',yaw_reward, pitch_reward,  mean_of_obj_locations)
            self.rewards_list.append(for_evaluation)
            # Store the transition in memory self.previous_state, self.previous_action, state, reward
            next_state = np.zeros(len(self.obs))
            next_state += self.obs
            if not flag_done:
                self.RL_controller.ERM.push(self.previous_state, self.previous_action, next_state, yaw_reward, pitch_reward)
            else: # list starting a new episode

                self.RL_controller.ERM.push_and_reward_modification(self.previous_state, self.previous_action, next_state, yaw_reward,pitch_reward)
                # I am not reseting the initial condition
                ##self.previous_action = []
                ##self.obs = np.zeros(self.RL_controller.history_buffer_size * self.RL_controller.num_of_states)
                print('episode finished!')

            if self.sample_counter % 100 == 0:
                #self.rewards_list
                name = 'VAR_rewards_his2_2'
                np.save(self.RL_controller.path_to_gathered_data +name, self.rewards_list)
                #np.save(self.RL_controller.path_to_gathered_data + str(self.RL_controller.num_of_experiments), self.RL_controller.ERM.memory)
                ##print('The overall performance: ',np.mean(np.array(self.rewards_list),0))

                np.save(self.RL_controller.path_to_gathered_data + name + '_traj', self.trajectory)
            if self.sample_counter % 10 == 0:
                self.RL_controller.learn()

            # the sample counter is used to update a random target for the PID controllers
            self.sample_counter += 1

        self.previous_state = np.zeros(len(self.obs))
        self.previous_state = self.previous_state + self.obs #TODO check this!!!! to be updated
        self.previous_action = self.get_action(yaw_ref, pitch_ref)
        #print('locations: ',mean_of_obj_locations)

    # for log purposes
    def pose_callback(self, msg):
        data_ = msg.pose.pose.position

        x, y, z = data_.x, data_.y, data_.z
        #print('pos: ', [x,y,z])
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
            pitch_rate = 0.05
        else:
            pitch_rate = -0.05
        return yaw_rate, pitch_rate, lin_vel

class DQN_approach:
    def __init__(self):
        self._init_hyperparameters()
        self.reset()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.history_buffer_size = 2
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
        self.BATCH_SIZE = 50
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4

    def reset(self):
        self.ERM = ReplayMemory(2000,self.GAMMA, config.MAX_EPISODE)
        self.steps_done = 0

    def select_action(self,state):

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if np.random.rand()>0.2: #sample > eps_threshold:
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
            final_next_states = torch.tensor(batch.next_state,device=self.device, dtype=torch.float32)
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
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()

        with torch.no_grad():
            outputs = self.target_net(final_next_states)
            next_state_values_yaw = outputs[0].max(1)[0]
            next_state_values_pitch = outputs[1].max(1)[0]
        # Compute the expected Q values
        yaw_loss = 0
        pitch_loss = 0
        # TD learning
        # for item1_yaw,item2_yaw,item3_yaw,item1_pitch,item2_pitch,item3_pitch in zip(
        #         batch.yaw_reward, next_state_values_yaw, state_action_values_yaw,
        #         batch.pitch_reward,next_state_values_pitch, state_action_values_pitch):
        #     if item1_yaw is None:
        #         expected_state_action_value_yaw = torch.tensor(-10.0, device=self.device, dtype=torch.float32)
        #     else:
        #         expected_state_action_value_yaw = (item2_yaw * self.GAMMA) + item1_yaw
        #
        #     if item1_pitch is None:
        #         expected_state_action_value_pitch = torch.tensor(-10.0, device=self.device, dtype=torch.float32)
        #     else:
        #         expected_state_action_value_pitch = (item2_pitch * self.GAMMA) + item1_pitch
        #
        #     yaw_loss += criterion(item3_yaw, expected_state_action_value_yaw.unsqueeze(0))
        #     pitch_loss += criterion(item3_pitch, expected_state_action_value_pitch.unsqueeze(0))
        ######### Monte Carlo
        # for item1_yaw,item3_yaw,item1_pitch,item3_pitch in zip(
        #         batch.yaw_reward, state_action_values_yaw,
        #         batch.pitch_reward, state_action_values_pitch):
        #
        #     print('hhhh: ',item3_yaw, item1_yaw)
        #     yaw_loss += criterion(item3_yaw, item1_yaw.unsqueeze(0))
        #     pitch_loss += criterion(item3_pitch, item1_pitch.unsqueeze(0))
        yaw_loss = criterion(yaw_reward_batch, state_action_values_yaw)
        pitch_loss = criterion(pitch_reward_batch, state_action_values_pitch)
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

    def __init__(self, capacity,gamma, max_episode = 100):
        self.memory = deque([],maxlen=capacity)
        self.episode_memory = deque([],maxlen=max_episode + 1) # we add one for the terminal state
        self.step = 0
        self.rewards_yaw = []
        self.rewards_pitch = []
        self.gamma = gamma
        self.max_episode = max_episode + 1
    def push(self, *args):
        """Save a transition"""
        new_data = Transition(*args)
        self.episode_memory.append(new_data)

        self.rewards_yaw.append(new_data.yaw_reward)
        self.rewards_pitch.append(new_data.pitch_reward)

        self.step += 1
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


    def push_and_reward_modification(self,*args):
        # Now modify the rewards accordingly!

        discount_reward_yaw = 0
        discount_reward_pitch = 0
        new_data = Transition(*args)
        #('state', 'action', 'next_state', 'yaw_reward', 'pitch_reward')
        if new_data.yaw_reward is None:
            new_data = Transition(new_data.state, new_data.action, new_data.next_state, -10.0, new_data.pitch_reward)

        if new_data.pitch_reward is None:
            new_data = Transition(new_data.state, new_data.action, new_data.next_state, new_data.yaw_reward, -10.0)

        self.episode_memory.append(new_data)

        self.rewards_yaw.append(new_data.yaw_reward)
        self.rewards_pitch.append(new_data.pitch_reward)

        self.step += 1

        count = 1
        for item1,item2 in zip(reversed(self.rewards_yaw), reversed(self.rewards_pitch)):
            discount_reward_yaw = item1 + discount_reward_yaw * self.gamma
            discount_reward_pitch = item2 + discount_reward_pitch * self.gamma
            self.episode_memory[-count] = Transition(self.episode_memory[-count].state, self.episode_memory[-count].action,
                                                                self.episode_memory[-count].next_state, discount_reward_yaw, self.episode_memory[-count].pitch_reward)

            self.episode_memory[-count] = Transition(self.episode_memory[-count].state, self.episode_memory[-count].action,
                                                                self.episode_memory[-count].next_state, self.episode_memory[-count].yaw_reward, discount_reward_pitch)
            self.memory.append(self.episode_memory[-count])
            count += 1


        print(self.episode_memory)
        self.step = 0
        self.rewards_yaw = []
        self.rewards_pitch = []
        self.episode_memory = deque([], maxlen=self.max_episode)
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
