import numpy as np
import matplotlib.pyplot as plt

colorlist = ["purple","blue", "orange", "green"]

np_2d_array_1 = np.load("saved_array_quat_1.npy")
np_2d_array_2 = np.load("saved_array_action_quad_1.npy")

print("Shape:",np_2d_array_1.shape)
print("Shape:",np_2d_array_2.shape)

for i in range(np_2d_array_1.shape[1]):
    plt.plot(np_2d_array_1[:,i], label=str(i), color = colorlist[i])

for i in range(np_2d_array_2.shape[1]):
    plt.plot(np_2d_array_2[:,i], label=str(i), color = colorlist[i])

plt.legend()
plt.show()
