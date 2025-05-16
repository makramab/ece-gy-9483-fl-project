# Federated Learning Project: Homogeneous and Heterogeneous Setup

This project explores **Federated Learning (FL)** using both **homogeneous** and **heterogeneous model setups**, trained over different datasets. It is designed for distributed deployment on cloud platforms with infrastructure automated using **Terraform** and **Ansible**.

## 🚀 Project Overview

The goal is to simulate federated learning where:

- **Clients** are trained on **non-shared private data**.
- Aggregation is done **centrally on a server**.
- Both **homogeneous (same architecture)** and **heterogeneous (different model architectures)** client configurations are explored.

Supported datasets:
- MNIST
- Fashion-MNIST
- CIFAR-10

Client models:
- BasicLinear
- Multi-Layer Perceptron (MLP)
- LeNet5

## 🧠 Features

- FL coordination and training using PyTorch
- Model heterogeneity support
- Quantization-based knowledge distillation
- Model accuracy and loss tracking
- Cloud infrastructure as code using Terraform
- Provisioning and software setup using Ansible
- Modular code for reproducibility

## 🎓 Directory Structure

```
.
├── clients/              # Client-side FL training scripts
├── server/               # Server-side aggregation logic
├── models/               # Model definitions (BasicLinear, MLP, LeNet5)
├── datasets/             # Dataset loading and partitioning
├── experiments/          # Experiment configurations and logs
├── terraform/            # Infrastructure setup using Terraform
├── ansible/              # Playbooks for provisioning (e.g. Python, PyTorch)
├── requirements.txt      # Python dependencies
└── README.md
```

## 📆 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/makramab/ece-gy-9483-fl-project.git
cd ece-gy-9483-fl-project
```

### 2. Infrastructure Provisioning
Provision GCE instances using Terraform:
```bash
cd terraform
terraform init
terraform apply
```

Set up clients and server using Ansible:
```bash
cd ../ansible
ansible-playbook setup.yml -i inventory.ini
```

### 3. Install Dependencies
On your local machine or inside your VM:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🎨 Running Experiments

### Homogeneous FL Setup
```bash
python run_homogeneous.py --model mlp --dataset mnist
```

### Heterogeneous FL Setup
```bash
python run_heterogeneous.py --models basiclinear mlp lenet --dataset fashionmnist
```

### CIFAR-10 Example
```bash
python run_heterogeneous.py --models mlp lenet --dataset cifar10
```

## 📊 Results
Results are logged per round, including:
- Local accuracy/loss
- Aggregated model accuracy
- Quantization error (for knowledge distillation setup)


