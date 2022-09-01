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
#from rlbench.backend.task import Task

from scipy.spatial.transform import Rotation


CURRENT_DIR = dirname(abspath(__file__))


cam_config = CameraConfig(rgb=True, depth=False, mask=False,
                          render_mode=RenderMode.OPENGL)
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.joint_positions = False
obs_config.gripper_pose = True
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
env = Environment(action_mode = act_mode, obs_config= obs_config,robot_setup = 'ur3baxter')

env.launch()

print(env.action_shape)
print(env.get_scene_data)

task_env = env.get_task(env._string_to_task('hd_ycb_grasp.py'))
print(task_env.get_name())

task_env.reset()
obs_init = task_env.get_observation().get_low_dim_data()
print(obs_init)

quat = np.array([0,1,0,0])
quat_norm = quat / np.linalg.norm(quat)

boundary_mins = [0.1, -0.2, 0.8]
boundary_maxs = [0.3, 0.2 , 0.95]

action_list = []
action_list.append(np.array([boundary_mins[0],boundary_mins[1],boundary_mins[2],quat_norm[0],quat_norm[1],quat_norm[2],quat_norm[3],1]))
action_list.append(np.array([boundary_mins[0],boundary_mins[1],boundary_maxs[2],quat_norm[0],quat_norm[1],quat_norm[2],quat_norm[3],1]))
action_list.append(np.array([boundary_mins[0],boundary_maxs[1],boundary_mins[2],quat_norm[0],quat_norm[1],quat_norm[2],quat_norm[3],1]))
action_list.append(np.array([boundary_mins[0],boundary_maxs[1],boundary_maxs[2],quat_norm[0],quat_norm[1],quat_norm[2],quat_norm[3],1]))
action_list.append(np.array([boundary_maxs[0],boundary_mins[1],boundary_mins[2],quat_norm[0],quat_norm[1],quat_norm[2],quat_norm[3],1]))
action_list.append(np.array([boundary_maxs[0],boundary_mins[1],boundary_maxs[2],quat_norm[0],quat_norm[1],quat_norm[2],quat_norm[3],1]))
action_list.append(np.array([boundary_maxs[0],boundary_maxs[1],boundary_mins[2],quat_norm[0],quat_norm[1],quat_norm[2],quat_norm[3],1]))
action_list.append(np.array([boundary_maxs[0],boundary_maxs[1],boundary_maxs[2],quat_norm[0],quat_norm[1],quat_norm[2],quat_norm[3],1]))

for _ in range(1):
    for action in action_list:
        observation, reward, done, info = task_env.step(action)

env.shutdown()
