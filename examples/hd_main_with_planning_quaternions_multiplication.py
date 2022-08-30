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


def quaternion_multiply(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=np.float64)

def euler_deg_to_quat(x,y,z):
    rot = Rotation.from_euler('xyz', [x, y, z], degrees=True)
    return rot.as_quat() # x,y,z,w

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

#quat = np.array([0,0,0,1]) # no rotation
quat = np.array([1,0,0,0]) # rotate 180 deg around the x-axis
#quat_rot = np.array([0,0,0.71,0.71]) # rotate 90 deg around the z-axis

quat_norm = quat / np.linalg.norm(quat)
action = np.array([0.3,-0.1,1.0,quat_norm[0],quat_norm[1],quat_norm[2],quat_norm[3],1])

quat_obs = []
quat_gt = []
xyz_obs = []
xyz_gt = []

for _ in range(10):
    print("#####################################")

    quat_norm = quaternion_multiply(quat_norm,euler_deg_to_quat(0,0,36))
    action = np.array([0.3,-0.1,1.0,quat_norm[0],quat_norm[1],quat_norm[2],quat_norm[3],1])
    print(action)
    observation, reward, done, info = task_env.step(action)
    # pose = arm_action_mode._pose_in_end_effector_frame(robot = task_env._robot, action = action[:7])

    print("x,y,z",observation.get_low_dim_data()[:3])
    print("quat",np.round(observation.get_low_dim_data()[3:], 2))


    xyz_obs.append(observation.get_low_dim_data()[:3].copy())
    xyz_gt.append(action[:3].copy())
    #quat_gt.append(pose[3:].copy())
    quat_obs.append(observation.get_low_dim_data()[3:].copy())
    quat_gt.append(action[3:7].copy())

env.shutdown()

np.save("np_xyz_obs",np.asarray(xyz_obs))
np.save("np_xyz_gt",np.asarray(xyz_gt))
np.save("np_quat_obs",np.asarray(quat_obs))
np.save("np_quat_gt",np.asarray(quat_gt))
