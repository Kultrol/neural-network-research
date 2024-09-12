from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(dataset_choice, batch_size):
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize images to 224x224 pixels
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_choice == 'mnist':
        train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    elif dataset_choice == 'fmnist':
        train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader
