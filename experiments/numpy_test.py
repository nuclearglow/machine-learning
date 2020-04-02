import numpy as np

# Array Element
a = np.array([[1, 2, 3], [4, 5, 6]])
print("Array", a, sep="\n")

# Array attrs
print("dimension", a.ndim, sep="\n")
print("size", a.size, sep="\n")

# Array creation
zeros = np.zeros((3, 3))
print("Zeroed Array", zeros, sep="\n")

ones = np.ones((3, 3))
print("Ones Array", ones, sep="\n")

empty = np.empty((3, 3))
print("Empty Array", empty, sep="\n")

ranged = np.arange(0, 16, 2)
print("Ranged Array", ranged, sep="\n")

random = np.random.rand(3, 3)
print("Random Array", random, sep="\n")

linear_spaced = np.linspace(1, 10, 20)
print("Linearspaced Array", linear_spaced, sep="\n")

circle_points = np.linspace(0, 2 * np.pi, 36)
print("circle point array", circle_points, sep="\n")

circle_func = np.sin(circle_points)
print("Circle func", circle_func, sep="\n")

# Array operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 3], [4, 5]])

print("Sum of all items in A", A.sum())
print("Sum of each column in A", A.sum(axis=0))
print("Sum of each row in A", A.sum(axis=1))
print("min item in A", A.min())
print("max item in A", A.max())

print("A*B = element-wise array mult", A * B, sep="\n")
print("A.dot(B) dotwise array mult", A.dot(B), sep="\n")

# Create a list object
a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Cast to numpy array
a = np.array(a)


# Arange generates arrays with numbers ranging from a to b (b excluded) with stepsize x
a = 0
b = 13
x = 0.5
c = np.arange(a, b, x)

# Generate zeros
z = np.zeros(2)
z = np.zeros((2, 3))

# Generate ones
o = np.ones(2)
o = np.ones((2, 3))

# Generate linspace (from 0 to 5 in 100 linearly spaced steps)
ls = np.linspace(0, 5, 100)

# Identity matrices
im = np.eye(4)

# Random numbers 5 by 5 matrix
unniform_randnums = np.random.rand(5, 5)
normal_randnums = np.random.randn(5, 5)
integer_randnums = np.random.randint(1, 101, (10, 5))  # from, to, dimensions

# Reshaping
newshape = integer_randnums.reshape(2, 25)

# Min/ max values
xmin = integer_randnums.min()
xmax = integer_randnums.max()

# Min/ max locations
xminloc = integer_randnums.argmin()
xmaxloc = integer_randnums.argmax()

# Dimensions / shape of array as tuple
array_shape = integer_randnums.shape

# Datatype of array
array_dtype = integer_randnums.dtype

# Standard deviation
array_std1 = integer_randnums.std(0)  # Along first dimension (rows)
array_std2 = integer_randnums.std(1)  # Along second dimension (columns)
array_std3 = integer_randnums.std()  # Whole array

# Indexing
arr = np.arange(0, 11)
arr[8]  # single value
arr[1:5]  # from to (exclusive)
arr[:6]  # Everything to index 6
arr[6:]  # Everything from (not including) index 6

# Numpy does not copy an array when slicing or when using "=", but references (Memory issues are the reason for that...)
arr = np.arange(1, 5, 0.5)
barr = arr
barr.fill(5)  # Now barr and arr is all fives...
carr = arr.copy()
carr.fill(4)  # Bbarr is still fives since carr is a true copy

# A 2d array...
arr_2d = [[5, 10, 15], [20, 25, 30], [35, 40, 45]]
arr_2d = np.array(arr_2d)
arr_2d[0][0]  # Double bracket notation single value
arr_2d[0]  # Double bracket notation first row
arr_2d[0, 0]  # Single bracket notation (better!)
arr_2d[0, :]  # Single bracket notation first row

# Advanced indexing using boolean vectors. Advanced indexing returns a true copy!
elements = arr_2d[np.mod(arr_2d, 10) == 0]
elements.fill(5)

# Element wise operations on arrays
a = np.arange(0, 10)
b = a + a
b = a - a
b = a + 5
b = a * a
b = a ** 2
b = np.sqrt(a)
