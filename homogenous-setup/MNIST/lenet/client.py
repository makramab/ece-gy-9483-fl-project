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
import time

# ========== LeNet Model ==========
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ========== Quantization Utils ==========
def quantize(tensor, num_bits=8):
    qmin = -2 ** (num_bits - 1)
    qmax = 2 ** (num_bits - 1) - 1
    scale = tensor.abs().max() / qmax
    quantized = torch.clamp((tensor / scale).round(), qmin, qmax)
    return quantized, scale

def dequantize(quantized_tensor, scale):
    return quantized_tensor * scale

# ========== Data ==========
def get_dataloaders(client_id):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    public_size = 10000
    private_size = len(full_train) - public_size
    private_data, public_data = random_split(full_train, [private_size, public_size])
    splits = [private_size // 3] * 3
    splits[2] += private_size % 3
    private_sets = random_split(private_data, splits)

    client_loader = DataLoader(private_sets[client_id - 1], batch_size=64, shuffle=True)
    public_loader = DataLoader(public_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)
    return client_loader, public_loader, test_loader

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total

# ========== Client Logic ==========
def run_client(client_id: int, server_ip: str):
    torch.manual_seed(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet5().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    distill_criterion = nn.KLDivLoss(reduction="batchmean")

    client_loader, public_loader, test_loader = get_dataloaders(client_id)

    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.IDENTITY, f"client{client_id}".encode())
    socket.connect(f"tcp://{server_ip}:5555")

    num_rounds = 10
    local_epochs = 1
    distill_temperature = 3

    for rnd in range(1, num_rounds + 1):
        print(f"\n[Client {client_id}] Round {rnd} - Training")
        model.train()
        for _ in range(local_epochs):
            for x, y in client_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

        acc = evaluate(model, test_loader, device)
        print(f"[Client {client_id}] Accuracy before distill: {acc*100:.2f}%")

        print(f"[Client {client_id}] Sending quantized logits to server")
        public_inputs, _ = next(iter(public_loader))
        public_inputs = public_inputs.to(device)
        logits = model(public_inputs).detach().cpu()

        quantized_logits, scale = quantize(logits, num_bits=8)

        msg = {
            "client_id": client_id,
            "quantized_logits": quantized_logits,
            "scale": scale,
            "round": rnd
        }
        socket.send_multipart([b"", pickle.dumps(msg)])

        # Receive teacher signal
        _, reply = socket.recv_multipart()
        data = pickle.loads(reply)
        teacher_logits = data["teacher_logits"]
        teacher_prob = F.softmax(teacher_logits.to(device) / distill_temperature, dim=1)

        model.train()
        optimizer.zero_grad()
        student_logits = model(public_inputs.to(device))
        student_log_prob = F.log_softmax(student_logits / distill_temperature, dim=1)
        loss_distill = distill_criterion(student_log_prob, teacher_prob)
        loss_distill.backward()
        optimizer.step()

        acc = evaluate(model, test_loader, device)
        print(f"[Client {client_id}] Accuracy after distill: {acc*100:.2f}%")

# ========== Entry Point ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True, help="1, 2, or 3")
    parser.add_argument("--server_ip", type=str, required=True, help="IP of the server")
    args = parser.parse_args()
    run_client(args.client_id, args.server_ip)
