import math

def windspd(u_ms, v_ms):
    return 1.94384 * math.sqrt(u_ms**2 + v_ms**2)

# u_ms, v_ms = input().replace('  ', ' ').split(' ')
# print(windspd(float(u_ms), float(v_ms)))

import matplotlib.pyplot as plt
import numpy as np

# Use Green's theorem to compute the area
# enclosed by the given contour.
def area(vs):
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return a

# Generate some test data.
delta = 0.01
x = np.arange(-3.1, 3.1, delta)
y = np.arange(-3.1, 3.1, delta)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)

# Plot the data
levels = [1.0,2.0,3.0,5.0]
cs = plt.contour(X,Y,r,levels=levels)
plt.clabel(cs, inline=1, fontsize=10)

# Get one of the contours from the plot.
for i in range(len(levels)):
    if i == len(levels)-1:
        breakpoint()
    contour = cs.collections[i]
    vs = contour.get_paths()[0].vertices
    # Compute area enclosed by vertices.
    a = area(vs)
    print("r = " + str(levels[i]) + ": a =" + str(a))

plt.show()
