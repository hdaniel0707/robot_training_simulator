import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trainname", default="supervised_02" ,help="Name of the training")
parser.add_argument("--dirsavelog", default="log" ,help="Dir for saving the logs")
args = parser.parse_args()

path = args.dirsavelog + "/" + args.trainname + "_test_loss.npy"
loss= np.load(path)

print("Shape:",loss.shape)
print("Mean:",loss.mean())

plt.plot(loss, label=args.trainname)
plt.title("Test Loss")
plt.legend()
plt.show()
