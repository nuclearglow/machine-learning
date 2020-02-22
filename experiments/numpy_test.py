import numpy

# Array Element
a = numpy.array([[1, 2, 3], [4, 5, 6]])
print("Array", a, sep="\n")

# Array attrs
print("dimension", a.ndim, sep="\n")
print("size", a.size, sep="\n")

# Array creation
zeros = numpy.zeros( (3,3) )
print("Zeroed Array", zeros, sep="\n")

ones = numpy.ones( (3,3) )
print("Ones Array", ones, sep="\n")

empty = numpy.empty( (3,3) )
print("Empty Array", empty, sep="\n")

ranged = numpy.arange( 0, 16, 2)
print("Ranged Array", ranged, sep="\n")

random = numpy.random.rand(3,3)
print("Random Array", random, sep="\n")

linear_spaced = numpy.linspace(1, 10, 20)
print("Linearspaced Array", linear_spaced, sep="\n")

circle_points = numpy.linspace(0, 2 * numpy.pi, 36)
print("circle point array", circle_points, sep="\n")

circle_func = numpy.sin(circle_points)
print("Circle func", circle_func, sep="\n")

# Array operations
A = numpy.array( [ [1, 2], [3, 4]])
B = numpy.array( [ [2, 3], [4, 5]])

print("Sum of all items in A", A.sum())
print("Sum of each column in A", A.sum(axis=0))
print("Sum of each row in A", A.sum(axis=1))
print("min item in A", A.min())
print("max item in A", A.max())

print("A*B = element-wise array mult", A*B, sep="\n")
print("A.dot(B) dotwise array mult", A.dot(B), sep="\n")
