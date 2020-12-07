import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import read_data
from collections import defaultdict
from shapely.geometry import Polygon

np.set_printoptions(precision=2)
some_points = np.array(read_data.main())
vor = Voronoi(some_points, furthest_site = False)
#fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='blue',line_width=1,line_alpha=0.5,point_size=1)
def remove_edges(vor):
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
if a dataset is sufficiently small all polygons can be constructed
with large datasets this takes far too long and is not efficient
'''
def small_scale_polygons(vor, diameter):
        points = vor.points
        ridge_points = vor.ridge_points
        vertices = vor.vertices
        ridge_vertices = vor.ridge_vertices
        centroid = points.mean(axis = 0)
        ridge_direction = defaultdict(list)

        for (p, q), rv in zip(ridge_points, ridge_vertices):
                u, v = sorted(rv)
                if u == -1:
                        tangent = points[q] - points[p]
                        normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
                        midpoint = points[[p, q]].mean(axis=0)
                        direction = np.sign(np.dot(midpoint - centroid, normal)) * normal
                        ridge_direction[p, v].append(direction)
                        ridge_direction[q, v].append(direction)
        for i, r in enumerate(vor.point_region):
                region = vor.regions[r]
                if -1 not in region:
                        yield Polygon(vor.vertices[region])
                        continue
                inf = region.index(-1)  
                j = region[(inf - 1) % len(region)] 
                k = region[(inf + 1) % len(region)] 
                if j == k: 
                        dir_j, dir_k = ridge_direction[i, j]
                else: 
                        dir_j, = ridge_direction[i, j]
                        dir_k, = ridge_direction[i, k]
                length = 2 * diameter / np.linalg.norm(dir_j + dir_k)
                finite_part = vor.vertices[region[inf + 1:] + region[:inf]]
                extra_edge = [vor.vertices[j] + dir_j * length,vor.vertices[k] + dir_k * length]
                yield Polygon(np.concatenate((finite_part, extra_edge)))

def small_scale_voronoi():
	points = np.random.randn(10,2)
	boundary = np.array([[-4, -2], [4,-2], [3,4], [1, 5], [-1, 4]])
	boundary_polygon = Polygon(boundary)
	vor = Voronoi(points)
	x, y = boundary.T
	plt.xlim(round(x.min() - 1), round(x.max() + 1))
	plt.ylim(round(y.min() - 1), round(y.max() + 1))
	plt.plot(*some_points.T,'b.')
	diameter = np.linalg.norm(boundary.ptp(axis=0))
	for poly in small_scale_polygons(vor, diameter):
		x, y = zip(*poly.intersection(boundary_polygon).exterior.coords)
		plt.plot(x,y,'b-')

def main():
	remove_edges(vor)
	small_scale_voronoi()

main()
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

