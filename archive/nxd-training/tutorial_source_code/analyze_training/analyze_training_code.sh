# Create the files needed
tee supported.py > /dev/null <<EOF
import torch
import torch_xla.core.xla_model as xm

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Linear(4,4)
        self.nl1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(4,2)
        self.nl2 = torch.nn.Tanh()

    def forward(self, x):
        x = self.nl1(self.layer1(x))
        return self.nl2(self.layer2(x))


def main():
    device = xm.xla_device()

    model = NN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    inp = torch.rand(4)
    target = torch.tensor([1,0])

    model.train()
    for epoch in range(2):
        optimizer.zero_grad()
        inp = inp.to(device)
        target = target.to(device)
        output = model(inp)
        loss = loss_fn(output,target)
        loss.backward()
        optimizer.step()
        xm.mark_step()

if __name__ == '__main__':
    main()
EOF

tee unsupported.py > /dev/null <<EOF
import torch
import torch_xla.core.xla_model as xm

class UnsupportedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y =  torch.fft.fft(x)
        x = x + 10
        return x * y


def main():
    device = xm.xla_device()

    model = UnsupportedModel().to(device)

    inp = torch.rand(4)

    model.train()
    for epoch in range(1):
        inp = inp.to(device)
        output = model(inp)

        xm.mark_step()

if __name__ == '__main__':
    main()
EOF

# Run analyze
neuron_parallel_compile --command analyze python supported.py

neuron_parallel_compile --command analyze python unsupported.py

neuron_parallel_compile --command analyze --analyze-verbosity 1 python unsupported.py