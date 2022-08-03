import os
import sys
from os.path import join, dirname, abspath, isfile
import math
import numpy as np

from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import ActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import ArmActionMode, JointPosition, EndEffectorPoseViaPlanning
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
joint_action_mode = EndEffectorPoseViaPlanning()
gripper_action_mode = Discrete()

act_mode = MoveArmThenGripper(joint_action_mode,gripper_action_mode)

env = Environment(action_mode = act_mode, obs_config= obs_config,robot_setup = 'ur3baxter')

env.launch()

print(env.action_shape)
print(env.get_scene_data)

task_env = env.get_task(env._string_to_task('hd_ycb_grasp.py'))
print(task_env.get_name())

task_env.reset()
obs_init = task_env.get_observation().get_low_dim_data()

task_env.reset()

print(obs_init[3:])
print(np.linalg.norm(obs_init[3:]))
quat = np.array([0,-0.966,0,-0.25])
print("Quat:",quat)
print(np.linalg.norm(quat))

quat_norm = quat / np.linalg.norm(quat)

action = np.concatenate((obs_init[:3],quat_norm,np.array([1]))).copy()
pos_x_list = [0.1,0.0,-0.1,0.0]
pos_z_list = [0.0,+0.1,0.0,-0.1]

quat_list = []
action_quat_list = []

for _ in range(10):
    for i in range (len(pos_x_list)):
        action[0] += pos_x_list[i]
        action[2] += pos_z_list[i]
        print(action)
        observation, reward, done, info = task_env.step(action)

        print("#####################################")
        print("x,y,z",observation.get_low_dim_data()[:3])
        print("quat",observation.get_low_dim_data()[3:])
        quat_list.append(observation.get_low_dim_data()[3:])
        action_quat_list.append(action[3:7])

env.shutdown()

np.save("saved_array_quat_1",np.asarray(quat_list))
np.save("saved_array_action_quad_1",np.asarray(action_quat_list))
