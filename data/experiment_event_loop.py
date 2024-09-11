import os

def experiment_event_loop():
    """
    Runs a series of experiments by iterating over various combinations of hyperparameters
    using a grid search approach. The function systematically varies the pruning percentage,
    batch size, and learning rate, and runs the main function for each combination.

    This function will:
    - Iterate over predefined sets of pruning percentages, batch sizes, and learning rates.
    - Cycle through learning rates using a round-robin method.
    - Save the results of each run in a directory structure based on the hyperparameters.

    Returns:
        None
    """
    # Print the selected device (e.g., CPU, CUDA, MPS)
    print(DEVICE)  # SHOW DEVICE

    # Define the total number of iterations to repeat the entire grid search
    total_iterations = 2

    # Initialize the index for learning rates to start at the first one
    lr_index = 0

    # Start the run counter from 1
    runs = 1

    # Loop until all iterations are complete
    while runs <= total_iterations:
        # Iterate over each pruning percentage
        for percentage in PERCENTAGES:
            # Iterate over each batch size
            for batch_size in ACCEPTABLE_BATCH_SIZES:
                # Iterate over each learning rate
                for lr in ACCEPTABLE_LR:
                    # Construct a filename that uniquely identifies the current run
                    filename = f"P%-{percentage:03}_BS-{batch_size:05}_LR-{lr}_run-{runs:03}"

                    # Construct a directory path for saving the results of this run
                    lr_dir = os.path.join(f"percentage_{percentage:03}",
                                          f"batch_size_{batch_size:05}",
                                          f"lr_{ACCEPTABLE_LR[lr_index]}",
                                          filename)

                    # Call the main function with the current hyperparameters and directory
                    main(percentage, batch_size, ACCEPTABLE_LR[lr_index], lr_dir)

                    # Move to the next learning rate in a round-robin fashion
                    lr_index = (lr_index + 1) % len(ACCEPTABLE_LR)

                    # You can optionally log or print progress information here

        # Increment the run counter after one full grid search iteration is complete
        runs += 1

    # You can optionally add any finalization steps here
    # For example, consolidating results or generating summary reports
