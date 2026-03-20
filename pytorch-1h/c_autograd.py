import torch
import torch.nn.functional as F
from torch.autograd import grad

# See computation_graph.png for the graph of this computation

# Data and parameters
y = torch.tensor([1.0])             # true label
x1 = torch.tensor([1.1])            # input feature
w1 = torch.tensor([2.2], requires_grad=True)
b  = torch.tensor([0.0], requires_grad=True)

# Forward pass before update
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)
print("loss before:", loss.item())

# Compute gradients dL/dw1 and dL/db
grad_L_w1, = grad(loss, w1, retain_graph=True) # unpack from the tuple
grad_L_b,  = grad(loss, b, retain_graph=True)

# Could have been done in one pass (avoid the need to retain the graph):
# grad_L_w1, grad_L_b = grad(loss, [w1, b])

# Or simply:
# loss.backward()
# grad_L_w1 = w1.grad
# grad_L_b = b.grad

print("grad_L_w1:", grad_L_w1.item())
print("grad_L_b:", grad_L_b.item())

# Gradient descent step: new_param = old_param - lr * gradient
learning_rate = 0.1
with torch.no_grad():
    w1 -= learning_rate * grad_L_w1
    b  -= learning_rate * grad_L_b

# Forward pass after update, to see if loss improved
z_new = x1 * w1 + b
a_new = torch.sigmoid(z_new)
loss_new = F.binary_cross_entropy(a_new, y)
print("loss after:", loss_new.item())
