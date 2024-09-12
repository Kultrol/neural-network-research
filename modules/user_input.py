def choose_dataset():
    while True:
        dataset_choice = input("Select dataset (mnist/fmnist): ").lower()
        if dataset_choice in ['mnist', 'fmnist']:
            return dataset_choice
        else:
            print("Invalid choice. Please select either 'mnist' or 'fmnist'.")

def set_batch_size():
    while True:
        try:
            batch_size = int(input("Enter batch size (Recommended: 32, 64, 128): "))
            if batch_size > 0 and batch_size <= 1024:
                return batch_size
            else:
                print("Please enter a batch size between 1 and 1024.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def set_learning_rate():
    while True:
        try:
            learning_rate = float(input("Enter learning rate: "))
            return learning_rate
        except ValueError:
            print("Invalid input. Please enter a valid float value.")

def confirm_settings(dataset, batch_size, learning_rate):
    print("\nPlease confirm your settings:")
    print(f"Dataset: {dataset}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    while True:
        confirm = input("Confirm settings (y/n): ").lower()
        if confirm in ['y', 'n']:
            return confirm == 'y'
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
