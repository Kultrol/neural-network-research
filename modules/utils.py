import os

def save_settings(settings_file, dataset, batch_size, learning_rate):
    with open(settings_file, "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")

def check_previous_training(results_dir):
    if os.path.exists(results_dir):
        settings_file = os.path.join(results_dir, "settings.txt")
        if os.path.exists(settings_file):
            while True:
                choice = input(f"Previous training found in {results_dir}. Do you want to continue or restart? (c/r): ").lower()
                if choice in ['c', 'r']:
                    return choice
                else:
                    print("Invalid input. Please enter 'c' to continue or 'r' to restart.")
    return None

def create_directories(prune_percentage, batch_size):
    # Create prune_layers directory
    prune_dir = "prune_layers"
    os.makedirs(prune_dir, exist_ok=True)

    # Create batch_size directory within prune_layers
    batch_size_dir = os.path.join(prune_dir, f"batch_size_{batch_size}")
    os.makedirs(batch_size_dir, exist_ok=True)

    # Create directory for the current prune percentage (increments of 5%)
    prune_percentage_dir = os.path.join(batch_size_dir, f"prune_{prune_percentage:03}%")
    os.makedirs(prune_percentage_dir, exist_ok=True)

    return prune_percentage_dir
