import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import model
import json

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.003
DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')

### Loading the dataset ###
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)) 
                                ])

train_set = datasets.MNIST(root='./data', train= True, transform= transform, download= True)
test_set = datasets.MNIST(root='./data', train= False, transform= transform, download= True)

train_loader = torch.utils.data.DataLoader(train_set, shuffle= True, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_set, shuffle= False, batch_size=BATCH_SIZE)

net = model.Net().to(DEVICE)

optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

train_losses = []
test_accuracies = []

for epoch in range(EPOCHS):
    net.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = net(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        ### For evaluation graph ###

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}: Training loss = {avg_loss:.4f}")

    ### For evaluation ###
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = net(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}: Test accuracy = {accuracy:.2f}%")


# Save metrics to a file
metrics = {
    'train_losses': train_losses,
    'test_accuracies': test_accuracies
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print('Training complete')

torch.save(net.state_dict(), 'trained_model.pth')
print("Model saved to 'trained_model.pth'.")