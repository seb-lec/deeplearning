import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += int(torch.sum(compare).item())
        total_examples += labels.size(0)

    return correct / total_examples

torch.manual_seed(123)

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
train_ds = ToyDataset(X_train, y_train)
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

# We initialize a model with two inputs and two outputs.
# That’s because the toy dataset has two input features and two class labels to predict.
model = NeuralNetwork(num_inputs=2, num_outputs=2)

# We used a stochastic gradient descent (SGD) optimizer with a learning rate (lr) of 0.5.
# The learning rate is a hyperparameter, meaning it’s a tunable setting that we have to experiment with based on observing the loss.
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# the number of epochs is another hyperparameter to choose.
# An epoch is one full pass through the training dataset.
# In this case, we will train for 3 epochs, meaning we will go through the training data 3 times.
num_epochs = 3

for epoch in range(num_epochs):

    # model.train() and model.eval() settings are used to put the model into a training and an evaluation mode.
    # This is necessary for components that behave differently during training and inference, such as dropout or batch normalization layers.
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)

        loss = F.cross_entropy(logits, labels) # Loss function which will apply the softmax function internally for efficiency and numerical stability reasons.

        optimizer.zero_grad() # clear the gradients from the previous step, otherwise they would accumulate and lead to incorrect updates of the model parameters.
        loss.backward() # calculate the gradients in the computation graph that PyTorch constructed in the background
        optimizer.step() # use the gradients to update the model parameters to minimize the loss.

        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    with torch.no_grad():
        outputs = model(X_train) # make predictions

    print("Outputs:", outputs)

    # To obtain the class membership probabilities, we can then use PyTorch’s softmax function, as follows:
    torch.set_printoptions(sci_mode=False)
    probas = torch.softmax(outputs, dim=1)
    print("Probabilities:", probas)

    # Get the predicted class labels by taking the argmax of the probabilities along the class dimension (dim=1):
    predicted_labels = torch.argmax(probas, dim=1) # also works with logits, because argmax is invariant to monotonic transformations like softmax
    print("Predicted labels:", predicted_labels)

    # check against the true labels
    print("True labels:", y_train)
    print("Predicted labels:", predicted_labels)
    print("Number of correct predictions:", torch.sum(predicted_labels == y_train))
    print("Is the model predicting correctly?", torch.equal(predicted_labels, y_train))
    print("Accuracy:", compute_accuracy(model, train_loader))

# Save the model:
# torch.save(model.state_dict(), "model.pth")
# Load the model:
# model = NeuralNetwork(2, 2) # needs to match the original model exactly
# model.load_state_dict(torch.load("model.pth"))

