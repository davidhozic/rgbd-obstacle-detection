from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.text import Annotation

import matplotlib.pyplot as plt
import numpy as np


ax: Axes3D = plt.figure().add_subplot(projection='3d')
fig = plt.gcf()

x_r = np.array([1, 0, 0])
y_r = np.array([0, 1, 0])
z_r = np.array([0, 0, 1])
ax.quiver(0, 0, 0, 1, 0, 0, color='r')
ax.quiver(0, 0, 0, 0, 1, 0, color='r')
ax.quiver(0, 0, 0, 0, 0, 1, color='r')
ax.text(1, 0, 0.1, "X", c='r')
ax.text(0, 1, 0.1, "Y", c='r')
ax.text(0, 0, 1.1, "Z", c='r')

ax.quiver(0.025, 0, 3.60, 0, 0, -1, color='b')
ax.quiver(0.025, 0, 3.60, 1, 0, 0, color='b')
ax.quiver(0.025, 0, 3.60, 0, -1, 0, color='b')
ax.text(1, 0, 3.75, "Z", c='b')
ax.text(0, -1, 3.75, "X", c='b')
ax.text(0, 0, 2.4, "Y", c='b')


ax.set_xlim3d([-1.5, 1.5])
ax.set_ylim3d([-1.5, 1.5])
ax.set_zlim3d([-1.5, 1.5])


class Annotation3D(Annotation):
    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)


red_arrow = Annotation3D(
    'Robot', (0, 0, 0),
    xytext=(30, 30),
    textcoords='offset points',
    arrowprops=dict(arrowstyle="-|>", color='red'),
    color='red'
)

blue_arrow = Annotation3D(
    'Kamera', (0, 0, 3.60),
    xytext=(30, 30),
    textcoords='offset points',
    arrowprops=dict(arrowstyle="-|>", color='blue'),
    color='blue'
)

ax.add_artist(red_arrow)
ax.add_artist(blue_arrow)
ax.axis('off')
plt.show()
