import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from qmnist import QMNIST

# Training set: 60,000 images (mirrors MNIST exactly)
print("Loading training data 1...")
qtrain = QMNIST(train=True, source_dir=r"../../qmnist")

# Testing set: first 10,000 images (mirrors classic MNIST test set)
print("Loading testing data 2...")
qtest = QMNIST(what='test10k', source_dir=r"../../qmnist")

# Loaders
batch_size = 32 # most commonly used default

train_loader = DataLoader(
    dataset=qtrain,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True # drop the last batch
)
    
test_loader = DataLoader(
    dataset=qtest,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# Define the model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(784, 128), # 784 input features (28x28 pixels)
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10), # 10 output classes (digits 0-9)
        )
    def forward(self, x):
        logits = self.layers(x)
        return logits

# Build the model, define the optimizer and the number of epochs
model = NeuralNetwork()

# Optimizers
# SGD	    One fixed learning rate for all weights 	        Simple, but needs careful lr tuning
# Adam	    Adapts the learning rate per weight automatically	Converges faster, more forgiving
# AdamW	    Adam + weight decay (regularisation)	            Best default for most modern networks
    
#optimizer = torch.optim.SGD(model.parameters(), lr=0.05) # Stochastic Gradient Descent optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001) # Adam optimizer (canonical default lr)

num_epochs = 10 # start small

# Train/Eval loop
for epoch in range(num_epochs):

    print(f"Epoch {epoch+1}/{num_epochs} | ", end="")

    # Start training
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, labels) # Loss function
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Start evaluation
    model.eval()
    label_ok = 0
    label_ko = 0
    val_loss = 0.0
    for batch_idx, (features, labels) in enumerate(test_loader):

        with torch.no_grad():
            logits = model(features) # make predictions
            val_loss += F.cross_entropy(logits, labels).item()
        
        predicted_label = torch.argmax(logits, dim=1)

        # Compare the predicted labels with the true labels
        label_ok += (predicted_label == labels).sum().item()
        label_ko += (predicted_label != labels).sum().item()

    print(f"Correctly classified: {label_ok}")
    print(f"Incorrectly classified: {label_ko}")
    accuracy = label_ok / (label_ok + label_ko)*100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Val Loss: {val_loss / len(test_loader):.2f}")
    # Loss measures confidence — how certain the model is about its predictions
    # Accuracy measures correctness — whether the top prediction was right or wrong
    # A model can be correct but increasingly overconfident
    # We should take the model that has the best combination of low loss and high accuracy,
    # not just the one with the highest accuracy.
    # Typically, we would save the model at the end of each epoch,
    # and then later select the one with the best validation performance for deployment.
    # trust val loss more than accuracy for judging whether to stop. Accuracy is noisy; loss is smoother.