import os
import sys
from os.path import join, dirname, abspath, isfile
import math
import numpy as np

from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import ActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import ArmActionMode, JointPosition, EndEffectorPoseViaPlanning, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import GripperActionMode, Discrete

from rlbench.task_environment import TaskEnvironment
from rlbench.backend.observation import Observation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from tqdm import tqdm
import time

torch.manual_seed(0)

from REINFORCE_Policy import Policy

CURRENT_DIR = dirname(abspath(__file__))

#############################################################################
# For compute you need to define the GPU and limit the CPU usage #############
# BEFORE IMPORTING PYTORCH
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 3 GPU
os.system("taskset -p -c 3-7 %d" % os.getpid()) #0-1-2 CPU
# For defining the GPUs: 'nvidia-msi'
# For defining the CPUs: 'top' and then press '1'
##############################################################################

print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

cam_config = CameraConfig(rgb=True, depth=False, mask=False,
                          render_mode=RenderMode.OPENGL)
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.joint_positions = False
obs_config.gripper_pose = True
obs_config.task_low_dim_state = True
obs_config.right_shoulder_camera = cam_config
obs_config.left_shoulder_camera = cam_config
obs_config.wrist_camera = cam_config
obs_config.front_camera = cam_config


#arm_action_mode = ArmActionMode()
#joint_action_mode = JointPosition()
#arm_action_mode = EndEffectorPoseViaPlanning(frame = 'end effector')
arm_action_mode = EndEffectorPoseViaPlanning()
#arm_action_mode = EndEffectorPoseViaIK()
#gripper_action_mode = GripperActionMode()
gripper_action_mode = Discrete()

act_mode = MoveArmThenGripper(arm_action_mode,gripper_action_mode)

#env = Environment(action_mode = act_mode, obs_config= obs_config,robot_setup = 'ur5baxter2')
env = Environment(action_mode = act_mode, obs_config= obs_config, headless = True, robot_setup = 'ur3baxter')

env.launch()

print(env.action_shape)
print(env.get_scene_data)

#task_env = env.get_task(env._string_to_task('hd_ycb_grasp.py'))
task_env = env.get_task(env._string_to_task('reach_target_no_distractors.py'))
#task_env = env.get_task(env._string_to_task('reach_target.py'))
print(task_env.get_name())


quat = np.array([0,1,0,0])
quat_norm = quat / np.linalg.norm(quat)

def print_data(observation, reward, done, info):
    print("----------------------------------")
    print(observation)
    print(observation.get_low_dim_data())
    print(observation.task_low_dim_state)
    print(reward)
    print(done)
    print(info)

def get_action(target_pose):
    return np.concatenate((target_pose, quat_norm, np.array([1])))

##########################
#TRAIN

max_t = 5
episode_num = 10

policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
loss_fn = torch.nn.MSELoss()
loss_list = []


for i in tqdm(range(episode_num), desc ="Epochs: "):
    task_env.reset()
    observation = task_env.get_observation()

    for j in range(max_t):

        inputs = torch.from_numpy(observation.task_low_dim_state.astype(np.float32)).to(device)
        labels = torch.from_numpy(observation.task_low_dim_state.astype(np.float32)).to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = policy(inputs)


        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        #print(loss.item())
        loss_list.append(loss.item())

print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
print(loss_list)

model_path = "model_checkpoints/supervised_01"
torch.save(policy.state_dict(), model_path)

env.shutdown()
