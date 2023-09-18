# Underwater scuba diver tracking using RL controllers
Implementation of paper: A comparison of RL-based and PID controllers for 6-DOF swimming robots: hybrid underwater object tracking

In this work, we present an exploration and assessment of employing a centralized deep Q-network (DQN) controller as a substitute for the prevalent use of PID controllers in the context of 6DOF swimming robots. Our primary focus centers on illustrating this transition with the specific case of underwater object tracking. DQN offers advantages such as data efficiency and off-policy learning, while remaining simpler to implement than other reinforcement learning methods. Given the absence of a dynamic model for our robot, we propose an RL agent to control this multi-input-multi-output (MIMO) system, where a centralized controller may offer more robust control than distinct PIDs. Our approach involves initially using classical controllers for safe exploration, then gradually shifting to DQN to take full control of the robot.

We divide the underwater tracking task into vision and control modules. We use established methods for vision-based tracking and introduce a centralized DQN controller. By transmitting bounding box data from the vision module to the control module, we enable adaptation to various objects and effortless vision system replacement. Furthermore, dealing with low-dimensional data facilitates cost-effective online learning for the controller. Our experiments, conducted within a Unity-based simulator, validate the effectiveness of a centralized RL agent over separated PID controllers, showcasing the applicability of our framework for training the underwater RL agent and improved performance compared to traditional control methods. 

Here is a video of our real-world test:

https://github.com/FARAZLOTFI/underwater_object_tracking/assets/44290848/9f5f66bb-d467-46de-9e1e-442ab84b6a18

This repo contains a ROS2 package that can be added to a workspace. 
Code structure:
*RL_network - The RL agent neural network model 
*DQN_controller_online_RL.py - Main code to run the algorithm 
*classic_controller_offpolicy.py - This is used to run PIDs to control the robot and gather dataset for offline RL
*gathering_dataset.py - This controls the robot manaully through keyboard where one can capture images, as well
*object_tracker.py - this is a the implemented vision module 

There is a "config" folder where config.py exists along with a guiding file named "topics in realworld". Changin this config file will allow us to use the same codes in realworld. 
