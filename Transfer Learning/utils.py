import matplotlib.pyplot as plt

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