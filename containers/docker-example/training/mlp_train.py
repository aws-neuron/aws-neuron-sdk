import os
import time
import torch
from model import MLP

from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# XLA imports
import torch_xla.core.xla_model as xm

# Global constants
EPOCHS = 4
WARMUP_STEPS = 2
BATCH_SIZE = 32

# Load MNIST train dataset
train_dataset = mnist.MNIST(root='./MNIST_DATA_train',
                            train=True, download=True, transform=ToTensor())

def main():
    # Prepare data loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

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
    model.train()
    for epoch in range(EPOCHS):
        start = time.time()
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
            if idx < WARMUP_STEPS: # skip warmup iterations
                start = time.time()

    # Compute statistics for the last epoch
    interval = idx - WARMUP_STEPS # skip warmup iterations
    throughput = interval / (time.time() - start)
    print("Train throughput (iter/sec): {}".format(throughput))
    print("Final loss is {:0.4f}".format(loss.detach().to('cpu')))

    # Save checkpoint for evaluation
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {'state_dict': model.state_dict()}
    # XLA: use xm.save instead of torch.save to ensure states are moved back to cpu
    # This can prevent "XRT memory handle not found" at end of test.py execution
    xm.save(checkpoint,'checkpoints/checkpoint.pt')

    print('----------End Training ---------------')

if __name__ == '__main__':
    main()