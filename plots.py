
import json
import matplotlib.pyplot as plt

# Load the metrics
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

train_losses = metrics['train_losses']
test_accuracies = metrics['test_accuracies']

# Plotting
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Over Epochs')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()