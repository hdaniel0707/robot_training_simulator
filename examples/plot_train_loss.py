import numpy as np
import matplotlib.pyplot as plt


training_name = "supervised_02"
path = "log/" + training_name + "_train_loss.npy"
training_loss= np.load(path)

#print(training_loss)
print("Shape:",training_loss.shape)

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

training_loss_mavg = movingaverage(training_loss,20)
print("Shape:",training_loss_mavg.shape)

plt.plot(training_loss, label=training_name)
plt.plot(training_loss_mavg, label=training_name + "_mavg")
plt.title("Training Loss")
plt.legend()
#plt.savefig("xyz.png")
plt.show()
