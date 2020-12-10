import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

np.set_printoptions(precision=2)
points = np.random.randn(10,2)
vor = Voronoi(points, furthest_site = False)

fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='blue',line_width=1,line_alpha=0.5,point_size=3)
graph = plt.figure(1, figsize= (4,3))

vertices = vor.vertices
ridge_vertices = vor.ridge_vertices
regions = vor.regions
point_region = vor.point_region

# Returns the Euclidean distance of two real-valued points	
def euclideanDistance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# Given an index for a point, returns the indices of the k 
# Voronoi nearest neighbors
def kNearestNeighbors(index, k):
	region = regions[point_region[index]]
	neighbors = []
	for i in range(len(region)):
		sharedEdge = [region[i], region[(i+1) % len(region)]]
		
		for j in range(len(points)):
			if j == index:
				continue
			if all(item in regions[point_region[j]] for item in sharedEdge):
				dist = euclideanDistance(points[index], points[j])
				neighbors.append([j, dist])
				break
	neighbors.sort(key=lambda x: x[1])
	return [neighbor[0] for neighbor in neighbors][:k]

# Returns the k-Distance of a given point.  This is the measure
# of how outlier-like the point is.
def kDistance(index, k):
	sum = 0
	for o in kNearestNeighbors(index, k):
		sum = sum + euclideanDistance(points[index], points[o])
	return sum / k

# Prints an ordered list of the points along with their Voronoi k-Distances
kDistances = [] 
for i in range(len(points)):
	kDistances.append([points[i], kDistance(i, 3)])
	kDistances.sort(key=lambda x: x[1])

print(kDistances)
plt.show()
