import torchvision
import torch.utils.data as data
from torchvision import transforms

def download_data(root='data', download=True, transform=None):
    """
    Downloads the FashionMNIST dataset (if not already downloaded) and applies transformations.

    Args:
        root (str): The root directory where the dataset will be stored.
        download (bool): If True, downloads the dataset if it's not already present.
        transform (torchvision.transforms.Compose): Transformations to apply to the data.

    Returns:
        tuple: A tuple containing:
            - train_dataset (torchvision.datasets.FashionMNIST): The training dataset.
            - test_dataset (torchvision.datasets.FashionMNIST): The testing dataset.
    """

    if transform is None:
        # Define default transformations if none are provided
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalizing with mean and std deviation
        ])

    # Download the FashionMNIST dataset with transformations applied
    train_dataset = torchvision.datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)

    return train_dataset, test_dataset

def data_handler(batch_size=64, root='data', download=True, transform=None):
    """
    Handles downloading and loading of the FashionMNIST dataset, returning DataLoader objects.

    Args:
        batch_size (int): The batch size to use for the data loaders.
        root (str): The root directory where the dataset will be stored.
        download (bool): If True, downloads the dataset if it's not already present.
        transform (torchvision.transforms.Compose): Transformations to apply to the data.

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
    """

    # Download datasets with the specified transformations
    train_dataset, test_dataset = download_data(root=root, download=download, transform=transform)

    # Create data loaders for training and testing datasets
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
