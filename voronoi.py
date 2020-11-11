import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

np.set_printoptions(precision=2)
points = np.random.randn(100,2)
vor = Voronoi(points, furthest_site = False)

fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='blue',line_width=1,line_alpha=0.5,point_size=3)

model1 = KMeans(n_clusters=9, init='random')
model1.fit(points)

print(model1.labels_)

plt.show()
