import numpy
from scipy import special, optimize
import matplotlib.pyplot as plt

f = lambda x: -special.jv(3, x)
sol = optimize.minimize(f, 1.0)

x = numpy.linspace(0, 10, 5000)

print("x", x)

plt.plot(x, special.jv(3, x), '-', sol.x, -sol.fun, 'o')
plt.savefig('scipy_plot_test.png', dpi=96)

