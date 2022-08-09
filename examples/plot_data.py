import numpy as np
import matplotlib.pyplot as plt

colorlist_xyz_gt = ["orange","green", "blue"]
colorlist_xyz_obs = ["moccasin","aquamarine","cornflowerblue"]
legend_xyz = ["x","y","z"]

colorlist_quat_gt = ["purple","blue", "orange", "green"]
colorlist_quat_obs = ["plum","cornflowerblue", "moccasin", "seagreen"]
legend_quat = ["qx","qy","qz","qw"]

np_xyz_gt= np.load("np_xyz_gt.npy")
np_xyz_obs = np.load("np_xyz_obs.npy")

np_quat_gt= np.load("np_quat_gt.npy")
np_quat_obs = np.load("np_quat_obs.npy")

print("Shape:",np_xyz_gt.shape)
print("Shape:",np_xyz_obs.shape)
print("Shape:",np_quat_gt.shape)
print("Shape:",np_quat_obs.shape)

for i in range(np_xyz_obs.shape[1]):
    plt.plot(np_xyz_obs[:,i], color = colorlist_xyz_obs[i], label=legend_xyz[i])

for i in range(np_xyz_gt.shape[1]):
    plt.plot(np_xyz_gt[:,i], color = colorlist_xyz_gt[i], label=legend_xyz[i])


plt.legend()
plt.savefig("xyz.png")
plt.show()

plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window

for i in range(np_quat_obs.shape[1]):
    plt.plot(np_quat_obs[:,i],color = colorlist_quat_obs[i], label=legend_quat[i])

for i in range(np_quat_gt.shape[1]):
    plt.plot(np_quat_gt[:,i],color = colorlist_quat_gt[i], label=legend_quat[i])

plt.legend()
plt.savefig("quat.png")
plt.show()
