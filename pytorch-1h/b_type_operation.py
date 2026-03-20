import torch

# *********
# Data type
# *********

# Python integers are 64-bit by default
tensor1d = torch.tensor([1, 2, 3])
print("tensor1d.dtype:", tensor1d.dtype) # torch.int64

# but if we create tensors from Python floats, PyTorch creates tensors with a 32-bit precision by default
floatvec = torch.tensor([1.0, 2.0, 3.0])
print("floatvec.dtype:", floatvec.dtype) # torch.float32

# That's good because GPU architectures are optimized for 32-bit computations

# to convert from 64-bit to 32-bit:
tensor1d_float32 = tensor1d.to(torch.float32)
print("tensor1d_float32.dtype:", tensor1d_float32.dtype) # torch.float32

# *****
# Shape
# *****

tensor2d = torch.tensor([[1, 2, 3],
                                      [4, 5, 6]])

print("tensor2d:", tensor2d) # tensor2d contents
print("tensor2d.shape:", tensor2d.shape) # torch.Size([2, 3])

# .shape returns [2, 3], which means that the tensor has 2 rows and 3 columns.
# To reshape the tensor into a 3 by 2 tensor, we can use the .reshape method:
tensor2d_3by2 = tensor2d.reshape(3, 2)
print("tensor2d_3by2:", tensor2d_3by2)
print("tensor2d_3by2.shape:", tensor2d_3by2.shape) # torch.Size([3, 2])

# Or use view
tensor2d_3by2_view = tensor2d.view(3, 2)
print("tensor2d_3by2_view:", tensor2d_3by2_view)
print("tensor2d_3by2_view.shape:", tensor2d_3by2_view.shape) # torch.Size([3, 2])

# we can use .T to transpose a tensor, which means flipping it across its diagonal.
# This is a common operation for matrices.
tensor2d_transposed = tensor2d.T
print("tensor2d_transposed:", tensor2d_transposed)
print("tensor2d_transposed.shape:", tensor2d_transposed.shape) # torch.Size([3, 2])

# To multiply two matrices in PyTorch, we use the .matmul method:
matrix_a = torch.tensor([[1, 2],
                         [3, 4]])
matrix_b = torch.tensor([[5, 6],
                         [7, 8]])
matrix_product = matrix_a.matmul(matrix_b)
print("matrix_product:", matrix_product)

# we can also adopt the @ operator, which accomplishes the same thing more compactly:
matrix_product_at = matrix_a @ matrix_b
print("matrix_product_at:", matrix_product_at)

# More tensor operations at https://pytorch.org/docs/stable/tensors.html

