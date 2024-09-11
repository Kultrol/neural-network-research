import torch
from utils.logger import Logger

def evaluation(test_data, model, device):
    """
    Evaluates a trained model on a test dataset and logs the results.

    Args:
        test_data (torch.utils.data.DataLoader): The DataLoader for the test dataset, which provides batches of input data and corresponding labels.
        model (torch.nn.Module): The trained model to be evaluated.
        device (torch.device): The device (CPU or GPU) on which the evaluation should be performed.

    Returns:
        None: This function does not return any value. It logs the average loss and accuracy of the model on the test dataset.

    The function performs the following steps:
    1. Sets the model to evaluation mode, disabling certain layers like dropout.
    2. Loops over the test data in batches, performing a forward pass to get predictions.
    3. Computes the loss and accuracy for each batch, accumulating these metrics.
    4. Averages the accumulated loss and accuracy over all batches.
    5. Logs the averaged loss and accuracy using a custom logger with a line counter.
    """

    # Get the total number of samples in the test dataset
    size = test_data.size

    # Get the number of batches in the test dataset
    number_of_batches = test_data.number_of_batches

    # Set the model to evaluation mode (disables dropout, etc.)
    model.evaluation()

    # Initialize variables to accumulate the total loss and number of correct predictions
    total_test_loss = 0.0
    total_test_correct = 0.0

    # Disable gradient calculations, since we don't need them for evaluation
    with torch.no_grad():
        # Iterate over batches of test data
        for input_data, label in test_data:
            # Move input data and labels to the specified device (CPU/GPU)
            input_data, label = input_data.to(device), label.to(device)

            # Perform a forward pass to get the model's predictions
            forward_pass = model(input_data)

            # Compute the loss for this batch and add it to the total test loss
            total_test_loss += model.criterion(forward_pass, label).item()

            # Count the number of correct predictions in this batch and add to the total correct count
            total_test_correct += (forward_pass.argmax(1) == label).type(torch.float).sum().item()

    # Calculate the average loss over all batches
    total_test_loss /= number_of_batches

    # Calculate the average accuracy over all samples
    total_test_correct /= size

    # Create an instance of the Logger class to handle logging
    logger = Logger()
    log_message = f"Test Loss: {total_test_loss:.6f}, Test Accuracy: {100 * total_test_correct:.2f}%"

    # Log the evaluation results, including the line number from the logger's line counter
    logger.log_to_console(log_message)
    logger.log_to_file(log_message)
