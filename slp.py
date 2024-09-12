import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.nn.utils.prune as prune
import torch.utils.data as data
import time
import numpy as np

# Setting the device
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

# Function to standardize data preprocessing
def get_data_loaders(dataset_choice, batch_size):
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize images to 224x224 pixels
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_choice == 'mnist':
        train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    elif dataset_choice == 'fmnist':
        train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset choice. Choose 'mnist' or 'fmnist'.")

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

# Function to count zeros in model parameters
def count_zeros(model):
    zero_params = 0
    total_params = 0
    for param in model.parameters():
        zero_params += torch.sum(param == 0).item()
        total_params += param.numel()
    print(f"Total parameters: {total_params}, Zero parameters: {zero_params}")

# Neural Network class adapted for DenseNet-121
class Neural_Network(nn.Module):

    def __init__(self, input_size, output_size, b_size, l_rate, percentage, prune_method='global', log_file='training_log.txt'):
        super().__init__()
        self.file = open(log_file, "w")
        self.percentage = percentage
        self.b_size = b_size
        self.l_rate = l_rate
        self.prune_method = prune_method

        # Initialize DenseNet-121
        self.model = models.densenet121(weights=None)
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier = nn.Linear(1024, output_size)

        self.model = self.model.to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.l_rate)

        self.ce_asy_list = [0]
        self.a_list = [0]
        self.b_list = [0]
        self.stop_counter = 0
        self.ces = []
        self.stop_training = False

        self.apply_pruning()

    def apply_pruning(self):
        if self.percentage > 0:
            if self.prune_method == 'global':
                # Global pruning
                parameters_to_prune = []
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        parameters_to_prune.append((module, 'weight'))
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.percentage / 100.0,
                )
            elif self.prune_method == 'layerwise':
                # Layer-wise pruning
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        prune.l1_unstructured(module, name='weight', amount=self.percentage / 100.0)
            else:
                raise ValueError("Prune method should be 'global' or 'layerwise'.")

    def forward(self, x):
        return self.model(x)

def train(dataloader, model, epoch):
    model.model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        model.optimizer.zero_grad()

        # Forward pass
        pred = model(X)
        loss = model.criterion(pred, y)

        # Backward pass and optimization
        loss.backward()
        model.optimizer.step()

        # Record loss and accuracy
        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        model.ces.append(loss.item())

        # Logging
        if batch % 10 == 0 or batch == num_batches - 1:
            current = batch * len(X)
            accuracy = 100 * correct / ((batch + 1) * model.b_size)
            avg_loss = total_loss / (batch + 1)
            model.file.write(f"{epoch}\t[{current:>5d}/{size:>5d}]\tLoss: {avg_loss:.6f}\tAccuracy: {accuracy:.2f}%\n")
            print(f"Epoch {epoch} [{current:>5d}/{size:>5d}] Loss: {avg_loss:.6f} Accuracy: {accuracy:.2f}%")

        # Memory management
        del X, y, pred, loss
        torch.cuda.empty_cache()

        # Stopping criterion
        if len(model.ces) >= 1000:
            model.stop_training = True
            break

    # Implementing the stopping criterion based on cross-entropy loss asymptote
    if len(model.ces) > 10:
        recent_losses = model.ces[-10:]
        if np.std(recent_losses) < 0.01:
            model.stop_training = True

def test(dataloader, model):
    model.model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += model.criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct * 100, test_loss

def main():
    # User input for hyperparameters
    dataset_choice = ''
    while dataset_choice not in ['mnist', 'fmnist']:
        dataset_choice = input("Select dataset (mnist/fmnist): ").lower()

    batch_size = 0
    while batch_size <= 0 or batch_size > 1024:
        try:
            batch_size = int(input("Enter batch size (1-1024): "))
        except ValueError:
            print("Please enter a valid integer.")

    learning_rate = 0
    while learning_rate <= 0:
        try:
            learning_rate = float(input("Enter learning rate (e.g., 0.001): "))
        except ValueError:
            print("Please enter a valid float.")

    # Confirm settings
    print(f"\nDataset: {dataset_choice}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    confirm = ''
    while confirm not in ['y', 'n']:
        confirm = input("Confirm settings? (y/n): ").lower()
    if confirm == 'n':
        print("Exiting.")
        return

    # Get data loaders
    train_loader, test_loader = get_data_loaders(dataset_choice, batch_size)

    # Initialize model
    input_size = 224 * 224  # Adjusted for resized images
    output_size = 10  # Number of classes
    prune_percentage = 0  # Set your desired pruning percentage here
    prune_method = 'global'  # 'global' or 'layerwise'

    # Create directories for logs
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_log.txt")

    model = Neural_Network(input_size, output_size, batch_size, learning_rate, prune_percentage, prune_method, log_file)

    # Training loop
    epochs = 100  # Set maximum number of epochs
    for epoch in range(1, epochs + 1):
        if model.stop_training:
            print("Stopping criterion met. Ending training.")
            break
        train(train_loader, model, epoch)
        accuracy, avg_loss = test(test_loader, model)

    # Count zeros in the model
    count_zeros(model.model)

    # Close the log file
    model.file.close()

    print(f"Training completed. Logs saved in {log_file}")

if __name__ == "__main__":
    main()
