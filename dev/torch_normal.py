import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
torch.manual_seed(0)

#############################################################################
# For compute you need to define the GPU and limit the CPU usage #############
# BEFORE IMPORTING PYTORCH
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 3 GPU
os.system("taskset -p -c 3-7 %d" % os.getpid()) #0-1-2 CPU
# For defining the GPUs: 'nvidia-msi'
# For defining the CPUs: 'top' and then press '1'
##############################################################################

print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Model to train
print("################")
print("Model to train: ")
mean = torch.tensor(0.0, requires_grad = True)
std = torch.tensor(1.0, requires_grad = False)
model = Normal(mean, std)
print(model)
print("Generate:")
output = model.sample([20])
print(output)

# Model GT
print("################")
print("Model GT: ")
mean_gt = torch.tensor(10.0)
std_gt = torch.tensor(1.0)
model_gt = Normal(mean_gt, std_gt)
print(model_gt)
print("Generate:")
output_gt = model_gt.sample([20])
print(output_gt)

# Optimizer and Loss
optimizer = torch.optim.Adam([mean, std],lr=0.1)

# Train
print("################")
print("Training ")
for i in range(300):
    output_gt = model_gt.sample([2000])
    model_logprobs = model.log_prob(output_gt)
    #print(model_logprobs)

    loss = -model_logprobs.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(i+1," iteration. Loss: ",loss.item())

# Model trained
print("################")
print("Model to train: ")
print(model)
print("Generate:")
output = model.sample([20])
print(output)
