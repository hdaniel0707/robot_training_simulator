import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trainname", default="supervised_xx" ,help="Name of the training")
parser.add_argument("--dirsavelog", default="log" ,help="Dir for saving the logs")
args = parser.parse_args()


path = args.dirsavelog + "/" + args.trainname + "_train_loss.npy"
train_loss= np.load(path)

#print(training_loss)
print("Shape:",train_loss.shape)

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

train_loss_mavg = movingaverage(train_loss,20)
print("Shape:",train_loss_mavg.shape)

plt.plot(train_loss, label=args.trainname)
plt.plot(train_loss_mavg, label=args.trainname + "_mavg")
plt.title("Training Loss")
plt.legend()
#plt.savefig("xyz.png")
plt.show()
