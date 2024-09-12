from modules.hardware import check_hardware
from modules.user_input import choose_dataset, set_batch_size, set_learning_rate, confirm_settings
from modules.data_loader import load_data
from modules.model_builder import build_densenet
from modules.pruning import prune_model
from modules.trainer import train_model, evaluate_model
from modules.utils import save_settings, check_previous_training, create_directories

import torch
import torch.optim as optim
import torch.nn as nn
import os

def main():
    # Check hardware
    device = check_hardware()

    # User input
    dataset_choice = choose_dataset()
    batch_size = set_batch_size()
    learning_rate = set_learning_rate()

    # Confirm settings
    if not confirm_settings(dataset_choice, batch_size, learning_rate):
        print("Exiting program.")
        return

    # Load data once
    train_loader, test_loader = load_data(dataset_choice, batch_size)

    for prune_percentage in range(0, 101, 5):
        print(f"Starting training with prune percentage: {prune_percentage}%")

        # Build the model
        model = build_densenet().to(device)

        # Apply pruning if needed
        if prune_percentage > 0:
            model = prune_model(model, prune_percentage)

        # Create directory structure for the current prune percentage and batch size
        results_dir = create_directories(prune_percentage, batch_size)

        # Check for previous training
        action = check_previous_training(results_dir)
        log_file_path = os.path.join(results_dir, "training_log.txt")
        model_checkpoint_path = os.path.join(results_dir, "model_checkpoint.pth")

        # Set optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        if action == 'r':
            # Delete old log and settings to restart training
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
            if os.path.exists(model_checkpoint_path):
                os.remove(model_checkpoint_path)
            print(f"Previous log and model checkpoint for prune percentage {prune_percentage}% deleted. Restarting training.")
            start_epoch = 1
        elif action == 'c':
            print(f"Continuing from previous training for prune percentage {prune_percentage}%.")
            if os.path.exists(model_checkpoint_path):
                checkpoint = torch.load(model_checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
                print(f"Resuming from epoch {start_epoch}")
            else:
                print(f"No checkpoint found at {model_checkpoint_path}, starting from scratch.")
                start_epoch = 1
        else:
            start_epoch = 1

        # Save settings to a file
        settings_file = os.path.join(results_dir, "settings.txt")
        save_settings(settings_file, dataset_choice, batch_size, learning_rate)

        # If starting fresh, write headers to the log file
        if not os.path.exists(log_file_path) or action == 'r':
            with open(log_file_path, "w") as log_file:
                log_file.write(f"{'Epoch':<5}\t{'[current/size]':<13}\t{'CE':<10}\t"
                               f"{'Accuracy(%)':<12}\t{'AVG_CE':<10}\t{'Batch':<5}\n")

        # Evaluate initial metrics before training
        print("Evaluating initial metrics before training...")
        try:
            initial_epoch = start_epoch - 1
            initial_accuracy = evaluate_model(model, device, test_loader, criterion)
            print(f"Initial accuracy: {initial_accuracy:.2f}%")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{initial_epoch:<5}\t[ 0000/{len(train_loader.dataset):<5d}]\t{'--':<10}\t"
                               f"{initial_accuracy:<12.3f}{'--':<10}\t{'0':<5}\n")
        except Exception as e:
            print(f"Error during initial evaluation: {e}")
            continue  # Skip to the next prune percentage

        # Training loop
        print("Starting training loop...")
        try:
            for epoch in range(start_epoch, 101):  # Train up to 100 epochs
                print(f"Starting epoch {epoch}...")
                accuracy = train_model(model, device, train_loader, optimizer, criterion, epoch, log_file_path)

                # Save model checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                torch.save(checkpoint, model_checkpoint_path)

                # Clear cache to free up memory
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during training: {e}")
            continue  # Skip to the next prune percentage

        print(f"Training completed for prune percentage {prune_percentage}%. Results saved to '{log_file_path}'.")

if __name__ == "__main__":
    main()

