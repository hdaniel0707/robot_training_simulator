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



# Create a rotation object from Euler angles specifying axes of rotation
rot = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)

# Convert to quaternions and print
rot_quat = rot.as_quat()
print(rot_quat)
action0 = np.concatenate((np.array([0.20,-0.10,1.0]),rot_quat,np.array([1])))
action0 = np.concatenate((obs_init[:3],rot_quat,np.array([1])))

observation, reward, done, info = task_env.step(action0)

action = action0.copy()

for _ in range(10):
    print("############################")
    for i in range(10):
            rot_quat2 = Rotation.from_euler('xyz', [i*60, i*15, i*45], degrees=True).as_quat()
            print(rot_quat2)
            action = np.concatenate((obs_init[:3],rot_quat2,np.array([1])))
            observation, reward, done, info = task_env.step(action)

            # print(observation.get_low_dim_data())
            # print(reward)
            # print(done)
            # print(info)

env.shutdown()
