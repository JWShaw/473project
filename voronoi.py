import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

np.set_printoptions(precision=2)
points = np.random.randn(20,2)
vor = Voronoi(points, furthest_site = False)

fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='blue',line_width=1,line_alpha=0.5,point_size=3)

model1 = KMeans(n_clusters=4, init='random')
model1.fit(points)

graph = plt.figure(1, figsize= (4,3))
lables = model1.labels_

vertices = vor.vertices
regions = vor.regions
region_idx = []

#finds regions that include sides outside of the diagram
for idx, region in enumerate(regions):
	if -1 in region:
		region_idx.append(idx)


to_remove = []
# finds the points corresponding to each outside region, likely outliers
for idx, points in enumerate(vor.point_region):
	if points in region_idx:
		to_remove.append(vor.points[idx])

print("points to remove are: ")
for removal in to_remove:
	print(removal)
'''
next_diagram = []
for point in vor.points:
	print(point)
	if point not in to_remove:
		next_diagram.append(point)
second_vor = Voronoi(next_diagram)
fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='red',line_width=1,line_alpha=0.5,    point_size=3)			
'''
plt.show()
