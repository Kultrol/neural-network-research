import os

def setup_experiment_directories():
    """
    Creates a directory structure based on the combinations of hyperparameters.
    This function will:
    - Create directories for each pruning percentage, batch size, and learning rate combination.
    - Ensure that the directory structure is created without errors, even if the directories already exist.

    After setting up the directories, the function will call the `experiment_event_loop`
    to run the experiments using the created directory structure.

    Returns:
        None
    """
    # Iterate over each pruning percentage to create the first level of directories
    for percentage in PERCENTAGES:
        # Format the directory name for the current pruning percentage
        percentage_dir = f"percentage_{percentage:03}"
        # Create the directory if it doesn't exist
        os.makedirs(percentage_dir, exist_ok=True)

        # Iterate over each batch size to create subdirectories under the current percentage directory
        for batch_size in ACCEPTABLE_BATCH_SIZES:
            # Format the directory name for the current batch size
            batch_size_dir = os.path.join(percentage_dir, f"batch_size_{batch_size:05}")
            # Create the directory if it doesn't exist
            os.makedirs(batch_size_dir, exist_ok=True)

            # Iterate over each learning rate to create subdirectories under the current batch size directory
            for lr in ACCEPTABLE_LR:
                # Format the directory name for the current learning rate
                lr_dir = os.path.join(batch_size_dir, f"lr_{lr}")
                # Create the directory if it doesn't exist
                os.makedirs(lr_dir, exist_ok=True)

    # After creating all necessary directories, start the experiment event loop
    experiment_event_loop()

