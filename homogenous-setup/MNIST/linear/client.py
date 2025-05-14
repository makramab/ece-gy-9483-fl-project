# client_quant_mnist_linear.py
import zmq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import argparse
import pickle

class BasicLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

def quantize(tensor, num_bits=8):
    qmin = -2 ** (num_bits - 1)
    qmax = 2 ** (num_bits - 1) - 1
    scale = tensor.abs().max() / qmax
    return torch.clamp((tensor / scale).round(), qmin, qmax), scale

def get_dataloaders(client_id):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    public_size = 10000
    private_size = len(full_train) - public_size
    private_data, public_data = random_split(full_train, [private_size, public_size])
    splits = [private_size // 3] * 3
    splits[2] += private_size % 3
    private_sets = random_split(private_data, splits)

    return (
        DataLoader(private_sets[client_id - 1], batch_size=64, shuffle=True),
        DataLoader(public_data, batch_size=64, shuffle=True),
        DataLoader(test, batch_size=64, shuffle=False)
    )

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

def run_client(client_id, server_ip):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicLinear().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    distill_criterion = nn.KLDivLoss(reduction="batchmean")

    client_loader, public_loader, test_loader = get_dataloaders(client_id)
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.IDENTITY, f"client{client_id}".encode())
    socket.connect(f"tcp://{server_ip}:5555")

    for rnd in range(1, 11):
        print(f"\n[Client {client_id}] Round {rnd}")
        model.train()
        for x, y in client_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, test_loader, device)
        print(f"[Client {client_id}] Accuracy before distill: {acc*100:.2f}%")

        x_pub, _ = next(iter(public_loader))
        x_pub = x_pub.to(device)
        logits = model(x_pub).detach().cpu()
        q_logits, scale = quantize(logits)

        socket.send_multipart([b"", pickle.dumps({"client_id": client_id, "quantized_logits": q_logits, "scale": scale, "round": rnd})])
        _, reply = socket.recv_multipart()
        teacher_logits = pickle.loads(reply)["teacher_logits"].to(device)

        optimizer.zero_grad()
        s_logits = model(x_pub)
        loss_distill = distill_criterion(F.log_softmax(s_logits / 3, dim=1), F.softmax(teacher_logits / 3, dim=1))
        loss_distill.backward()
        optimizer.step()

        acc = evaluate(model, test_loader, device)
        print(f"[Client {client_id}] Accuracy after distill: {acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--server_ip", type=str, required=True)
    args = parser.parse_args()
    run_client(args.client_id, args.server_ip)
