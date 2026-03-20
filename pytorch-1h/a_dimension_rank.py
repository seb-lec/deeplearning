import torch

# This script summarizes the difference between
#  - mathematical dimension (like "3D vector")
#  - tensor rank / array dimensions in PyTorch

# A scalar is just a single number
# It has no axes and needs no indices to access it
scalar = torch.tensor(5.0)
print("scalar:", scalar)
print("scalar.shape:", scalar.shape)  # torch.Size([]) means 0D, rank 0
print()


# A vector is a 1D tensor
# A "3D vector" in math usually means a vector with 3 components: [x, y, z]
# In PyTorch this is represented as a 1D tensor of length 3
vector_3d = torch.tensor([1.0, 2.0, 3.0])
print("3D vector (3 components):", vector_3d)
print("vector_3d.shape:", vector_3d.shape)  # torch.Size([3]) means 1D, rank 1

# You access elements with a single index
print("vector_3d[0]:", vector_3d[0])  # first component
print()


# A matrix is a 2D tensor
# It has two axes: rows and columns
matrix = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
print("matrix:\n", matrix)
print("matrix.shape:", matrix.shape)  # torch.Size([2, 3]) means 2D, rank 2

# You access elements with two indices: row, column
print("matrix[0, 1]:", matrix[0, 1])  # first row, second column
print()


# A "3D tensor" in the array sense means a tensor with three axes
# For example, this tensor has shape (2, 3, 4) and is rank 3
# You need three indices to get a single element
tensor_3d = torch.randn(2, 3, 4)
print("3D tensor (rank 3):")
print("tensor_3d.shape:", tensor_3d.shape)  # torch.Size([2, 3, 4])

# You access elements with three indices
print("tensor_3d[0, 1, 2]:", tensor_3d[0, 1, 2])
print()


# Summary:
#  - A "3D vector" in math is a vector with 3 components, like [x, y, z]
#    In PyTorch this is a 1D tensor with shape (3,)
#    It is rank 1, even though the vector lives in 3-dimensional space
#
#  - A "3D tensor" in PyTorch is something with three axes (rank 3)
#    For example, shape (2, 3, 4) means you need three indices to access elements
