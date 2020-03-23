import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep

x = numpy.linspace(0, 10, 10)
y = numpy.sin(x)
spl = splrep(x, y)

x2 = numpy.linspace(0, 10, 200)
y2 = splev(x2, spl)
plt.plot(x, y, "o", x2, y2)
plt.show()

# import numpy
# from scipy import special, optimize
# import matplotlib.pyplot as plt

# f = lambda x: -special.jv(3, x)
# sol = optimize.minimize(f, 1.0)

# x = numpy.linspace(0, 10, 5000)

# print("x", x)

# plt.plot(x, special.jv(3, x), '-', sol.x, -sol.fun, 'o')
# plt.savefig('scipy_plot_test.png', dpi=96)

# https://github.com/justjanne/powerline-go
