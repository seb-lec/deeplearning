import torch
"""
A multi-layer neural network model using PyTorch.
This class defines a feedforward neural network with two hidden layers.
It inherits from torch.nn.Module, which provides the base functionality
for building and training neural network models in PyTorch.
Architecture:
    - Input layer: num_inputs units
    - Hidden layer 1: 30 units with ReLU activation
    - Hidden layer 2: 20 units with ReLU activation
    - Output layer: num_outputs units (linear, no activation)
Attributes:
    layers (torch.nn.Sequential): A sequential container of linear layers
        and activation functions that compose the network.
Args:
    num_inputs (int): The number of input features.
    num_outputs (int): The number of output units.
Methods:
    forward(x): Performs a forward pass through the network.
"""
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        # Using Sequential is not required, but it can make our life easier if we have a series of layers
        # that we want to execute in a specific order, as is the case here.
        # This way, after instantiating self.layers = Sequential(...) in the __init__ constructor,
        # we just have to call the self.layers instead of calling each layer individually
        # in the NeuralNetwork’s forward method.
        self.layers = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


torch.manual_seed(123) # to be able to reproduce the same random numbers (for the weights initialization)
# Randomness is used in the initialization of the weights of the layers, because it allows the model to break symmetry and learn better.
# If all the weights were initialized to the same value, then all the neurons in a layer would learn the same features and the model would not be able to learn complex patterns in the data.

model = NeuralNetwork(num_inputs=10, num_outputs=3)
print(model)

num_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print("Total number of trainable model parameters:", num_params)
# The trainable parameters are contained in the torch.nn.Linear layers.
# A linear layer multiplies the inputs with a weight matrix and adds a bias vector.
# This is sometimes also referred to as a feedforward or fully connected layer.

print("Shape of the first layer's weights:", model.layers[0].weight.shape)
print("Weights of the first layer:", model.layers[0].weight)
print("Shape of the first layer's bias:", model.layers[0].bias.shape)
print("Bias of the first layer:", model.layers[0].bias)


x = torch.randn(32, 10) # batch of 32 samples with 10 features
output = model(x)
print("Shape of the output:", output.shape) # torch.Size([32, 3])
print("Output of the model:", output) # tensor([...], grad_fn=<AddmmBackward0>)

# AddmmBackward0 means that the output was computed by a matrix multiplication (the "mm" part)
# followed by an addition (the "add" part).


# When we use a model for inference rather than training,
# it is a best practice to use the torch.no_grad() context manager.
# This tells PyTorch that it doesn’t need to keep track of the gradients,
# which can result in significant savings in memory and computation.
with torch.no_grad():
    output_no_grad = model(x)

# Model output

# PyTorch does not automatically return probabilities from the model, because:
# For training, loss functions want logits.
# For inference, sometimes you only need the argmax (the most likely class) and not the probabilities.

# Model returns logits (scores).
# - Loss functions like `CrossEntropyLoss` take logits and internally apply softmax + log + NLL.
# - “Class-membership probabilities” are what you get when you explicitly apply `softmax` to the logits:
#   - one probability per class
#   - in `[0, 1]`
#   - sum to 1 per example.

