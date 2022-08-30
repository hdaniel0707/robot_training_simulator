from scipy.spatial.transform import Rotation
import numpy as np

# Create a rotation object from Euler angles specifying axes of rotation
rot = Rotation.from_euler('xyz', [0, 0, 90], degrees=True)

# Convert to quaternions and print
rot_quat = rot.as_quat() # x,y,z,w

print(np.round(rot_quat,2))
