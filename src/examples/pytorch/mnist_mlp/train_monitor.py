import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import mnist
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# XLA imports
import torch_xla.core.xla_model as xm

# Declare 3-layer MLP for MNIST dataset
class MLP(nn.Module):
    def __init__(self, input_size = 28 * 28, output_size = 10, layers = [120, 84]):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Load MNIST train dataset
train_dataset = mnist.MNIST(root='./MNIST_DATA_train', \
                            train=True, download=True, transform=ToTensor())

def main():
    # Prepare data loader
    train_loader = DataLoader(train_dataset, batch_size=32)

    # Fix the random number generator seeds for reproducibility
    torch.manual_seed(0)

    # XLA: Specify XLA device (defaults to a NeuronCore on Trn1 instance)
    device = 'xla'

    # Move model to device and declare optimizer and loss function
    model = MLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.NLLLoss()

    # Run the training loop
    print('----------Training ---------------')
    for run in range(0, 1000):
        print(f'Run {run}')
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            optimizer.zero_grad()
            train_x = train_x.view(train_x.size(0), -1)
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            output = model(train_x)
            loss = loss_fn(output, train_label)
            loss.backward()
            optimizer.step()
            xm.mark_step() # XLA: collect ops and run them in XLA runtime
            if idx < 2: # skip warmup iterations
                start = time.time()

    # Save checkpoint for evaluation
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {'state_dict': model.state_dict()}
    # XLA: use xm.save instead of torch.save to ensure states are moved back to cpu
    # This can prevent "XRT memory handle not found" at end of test.py execution
    xm.save(checkpoint,'checkpoints/checkpoint.pt')

    print('----------End Training ---------------')