import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

import flwr as fl
from flwr.client import NumPyClient
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# Flower server address (replace with actual server IP)
SERVER_ADDRESS = "<server_ip>:8080"

# 𝚃ransform pipeline: PIL → Tensor + normalize
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Custom collate_fn to apply transform and stack into batches
def collate_fn(batch):
    images = [transform(example["img"]) for example in batch]
    labels = [example["label"] for example in batch]
    return torch.stack(images), torch.tensor(labels)


# Define model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Convert between PyTorch state_dict and Flower’s parameters
def get_parameters(model: nn.Module, config):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model: nn.Module, parameters):
    state_dict = model.state_dict()
    for (key, _), param in zip(state_dict.items(), parameters):
        state_dict[key] = torch.tensor(param)
    model.load_state_dict(state_dict)


# Local train & test routines
def train(net, loader):
    net.train()
    optimizer = optim.Adam(net.parameters())
    loss_fn = nn.CrossEntropyLoss()
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = net(images)
        loss_fn(outputs, labels).backward()
        optimizer.step()

def test(net, loader):
    net.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = net(images)
            total_loss += loss_fn(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


# Flower client
class FLClient(NumPyClient):
    def __init__(self, cid: str):
        self.cid = int(cid)

        # Partition CIFAR-10 train split into 2 IID shards
        fds = FederatedDataset(
            dataset="cifar10",
            partitioners={"train": IidPartitioner(num_partitions=2)},
            shuffle=True,
            seed=42,
        )

        # Load this client’s training partition
        train_ds = fds.load_partition(partition_id=self.cid)
        self.trainloader = DataLoader(
            train_ds,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn,
        )

        # Load the shared test split
        test_ds = fds.load_split("test")
        self.testloader = DataLoader(
            test_ds,
            batch_size=32,
            collate_fn=collate_fn,
        )

        # Initialize the model
        self.net = Net()

    # Accept the extra `config` parameter
    def get_parameters(self, config):
        return get_parameters(self.net, config)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader)
        return get_parameters(self.net, config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 flower_client.py <client_id>")
        sys.exit(1)

    client_id = sys.argv[1]
    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=FLClient(client_id),
    )

