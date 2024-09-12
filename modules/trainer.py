import torch

def evaluate_model(model, device, data_loader, criterion):
    model.eval()
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total_samples += target.size(0)
    accuracy = 100.0 * correct / total_samples
    return accuracy


def train_model(model, device, train_loader, optimizer, criterion, epoch, log_file_path):
    model.train()
    running_correct = 0
    total_samples = 0
    cumulative_loss = 0.0
    total_batches = len(train_loader)
    dataset_size = len(train_loader.dataset)

    # Open the log file in append mode
    with open(log_file_path, "a") as log_file:
        for batch_idx, (data, target) in enumerate(train_loader):
            current_samples = batch_idx * data.size(0) + data.size(0)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate batch accuracy
            _, predicted = torch.max(output, 1)
            batch_correct = (predicted == target).sum().item()
            batch_accuracy = 100.0 * batch_correct / target.size(0)

            # Update running totals
            running_correct += batch_correct
            total_samples += target.size(0)
            cumulative_loss += loss.item() * target.size(0)  # Multiply by batch size

            # Calculate average cross-entropy loss up to current batch
            avg_ce = cumulative_loss / total_samples

            # Format and write to log file
            log_file.write(f"{epoch:<5}\t[{current_samples:>5d}/{dataset_size:<5d}]\t"
                           f"{loss.item():<10.6f}\t{batch_accuracy:<12.3f}"
                           f"{avg_ce:<10.6f}\t{batch_idx + 1:<5}\n")
            # Flush the log file to ensure data is written to disk
            log_file.flush()

            # Delete variables and clear cache to free up memory
            del data, target, output, loss, predicted
            torch.cuda.empty_cache()

    # Calculate and display epoch-level metrics
    epoch_accuracy = 100.0 * running_correct / total_samples
    avg_epoch_loss = cumulative_loss / total_samples
    print(f"Epoch {epoch} completed - Accuracy: {epoch_accuracy:.2f}%, Avg Loss: {avg_epoch_loss:.4f}")

    return epoch_accuracy
