import torch
import numpy as np

def config_handler(seed=None, enable_mixed_precision=False):
    """
    Configures PyTorch based on the available hardware and user preferences.

    Args:
        seed (int, optional): Seed for random number generators to ensure reproducibility.
        enable_mixed_precision (bool, optional): If True, enable mixed precision training for GPUs.

    Returns:
        device (torch.device): The selected device (CPU, CUDA, or MPS).
    """

    # Set print options for debugging
    torch.set_printoptions(threshold=100000000000)

    # Optionally set a seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # Ensures reproducibility at the cost of performance
        torch.backends.cudnn.benchmark = False

    # Select the best available device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Print the selected device
    print(f"Selected device: {device}")

    # Enable CuDNN benchmark for CUDA devices to optimize convolutional layers
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Optionally enable mixed precision for GPUs
    if enable_mixed_precision and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed precision enabled.")
    else:
        scaler = None

    return device, scaler
