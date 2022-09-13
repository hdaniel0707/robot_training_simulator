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
torch.manual_seed(0)

from REINFORCE_Policy_dist import Policy

from tqdm import tqdm
import time

CURRENT_DIR = dirname(abspath(__file__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trainname", default="rl_reinforce_01" ,help="Name of the training")
parser.add_argument("--dim", type=int, default=1 , choices=[1, 2, 3], help="Define the dimensions of the problem")
parser.add_argument("--dirsave", default="model_checkpoints" ,help="Dir for saving the models")
parser.add_argument("--dirsavelog", default="log" ,help="Dir for saving the logs")
parser.add_argument("--iternum", type=int, default=20 ,help="Define the number of iterations")
parser.add_argument("--batchsize", type=int, default=100 ,help="Define the number of iterations")
parser.add_argument("--maxt", type=int, default=1 ,help="Define the horizon of an episode")
parser.add_argument("--gpu", type=int, default=0 ,help="Define the GPU number (only one)")
parser.add_argument("--cpumin", type=int, default=3 ,help="Define the lower value of CPU interval")
parser.add_argument("--cpumax", type=int, default=7 ,help="Define the upper value of CPU interval")
args = parser.parse_args()

boundary_mins = [0.1, -0.2, 0.76]
boundary_maxs = [0.35, 0.2 , 0.96]

#############################################################################
# For compute you need to define the GPU and limit the CPU usage #############
# BEFORE IMPORTING PYTORCH
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) # 3 GPU
os.system("taskset -p -c "+str(args.cpumin)+"-"+str(args.cpumax)+" %d" % os.getpid()) #0-1-2 CPU
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

arm_action_mode = EndEffectorPoseViaPlanning()
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

#action = np.array([0.3,-0.1,1.0,quat_norm[0],quat_norm[1],quat_norm[2],quat_norm[3],1])

def print_data(observation, reward, done, info):
    print("----------------------------------")
    print(observation)
    print(observation.get_low_dim_data())
    print(observation.task_low_dim_state)
    print(reward)
    print(done)
    print(info)

def action_to_action_robot(action, obs ):
    action_np = action.clone().cpu().detach().numpy().flatten()
    #print(action_np)
    action_np_clipped = np.clip(action_np, boundary_mins[:args.dim], boundary_maxs[:args.dim])
    if args.dim == 3:
        return np.concatenate((action_np_clipped, quat_norm, np.array([1])))
    elif args.dim == 2:
        return np.concatenate((action_np_clipped, np.array([obs[2]]) , quat_norm, np.array([1])))
    elif args.dim == 1:
        # print(action_np_clipped)
        # print(np.array([obs[1:3]]))
        return np.concatenate((action_np_clipped, np.array(obs[1:3]) , quat_norm, np.array([1])))

policy = Policy(state_size=args.dim, action_size=args.dim, boundary_mins=boundary_mins, boundary_maxs=boundary_maxs).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
loss_fn = torch.nn.MSELoss()
loss_list = []

for _ in tqdm(range(args.iternum), desc ="Iterations: "):

    states = []
    actions = []
    rewards = []

    for i in tqdm(range(args.batchsize), desc ="Collecting batch samples: "):

        task_env.reset()
        observation = task_env.get_observation()

        for j in range(args.maxt):

            state = torch.from_numpy(observation.task_low_dim_state[:args.dim].astype(np.float32)).to(device)
            states.append(state)

            pi = policy(state)
            action = pi.sample()
            actions.append(action)
            action_robot = action_to_action_robot(action, observation.task_low_dim_state)
            observation, reward, done, info = task_env.step(action_robot)
            rewards.append(torch.tensor(reward,device=device))

            if done:
                break

    # print(states)
    # print(actions)
    # print(rewards)

    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    rewards = rewards.view(-1,1)
    #rewards.to(device)

    # print(states)
    # print(actions)
    # print(rewards)

    # update policy
    pi = policy(states)
    logprobs = pi.log_prob(actions)
    loss = - (logprobs * rewards).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #print(loss.item())
    loss_list.append(loss.item())


print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
print(loss_list)

model_path = args.dirsave + "/" + args.trainname
torch.save(policy.state_dict(), model_path)

logpath = args.dirsavelog + "/" + args.trainname + "_train_loss"
np.save(logpath,np.asarray(loss_list))

env.shutdown()
