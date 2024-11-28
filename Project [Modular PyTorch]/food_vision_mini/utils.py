"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
import json
import matplotlib.pyplot as plt



def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
    either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth"
            )
    """
    
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)
    

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    
    return total_time

# Save results to a JSON file
def save_results_to_json(results: dict, filepath: str):
    """Saves the results dictionary to a JSON file.

    Args:
        results (dict): The dictionary containing training/testing metrics.
        filepath (str): The file path where the JSON file will be saved.
    """
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save the training result
    print(f"[INFO] Saving training results to: {filepath}")

# Load results from a JSON file
def load_results_from_json(filepath: str) -> dict:
    """Loads the results dictionary from a JSON file.

    Args:
        filepath (str): The file path to the JSON file to be read.

    Returns:
        dict: The dictionary containing training/testing metrics.
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def plot_training_history(epochs, train_losses, train_accuracies, test_losses, test_accuracies):
    """
    Plots training history for loss and accuracy with an X-axis interval of 5 epochs.
    """
    # Create figure
    plt.figure(figsize=(12, 6))

    # Define X-axis ticks at intervals of 5
    x_ticks = range(1, epochs + 1, 5)  # Interval of 5

    # Subplot for training and testing loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', linewidth=2.5, linestyle='-', color='blue')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', linewidth=2.5, linestyle='--', color='orange')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Testing Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(x_ticks, fontsize=10)
    plt.yticks(fontsize=10)

    # Subplot for training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', linewidth=2.5, linestyle='-', color='green')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', linewidth=2.5, linestyle='--', color='red')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Testing Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(x_ticks, fontsize=10)
    plt.yticks(fontsize=10)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show plot
    plt.show()    