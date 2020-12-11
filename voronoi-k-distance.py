import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import math
import sys
import read_data
from sklearn.cluster import KMeans
from sklearn import datasets
from scipy.spatial import Voronoi, voronoi_plot_2d
from statistics import median

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
	all_neighbors = [n[0] for n in neighbors]
	neighbor_ids.append(list(all_neighbors))
	return [neighbor[0] for neighbor in neighbors][:k]

# Returns the k-Distance of a given point.  This is the measure
# of how outlier-like the point is.
def kDistance(index, k):
	sum = 0
	for o in kNearestNeighbors(index, k):
		sum = sum + euclideanDistance(points[index], points[o])
	return sum / k

def region_perimeter(index):
	region = regions[point_region[index]]
	result = 0
	for i in range(len(region)):
		if region[i] == -1:
			return -1
		result = result + euclideanDistance(vertices[region[i]], \
				 vertices[region[(i + 1) % len(region)]])
	return result

def getOutlierValues(kdistances):
	outlier_vals = []
	outlier_class = []
	max_val = max(kdistances)
	min_val = min(kdistances)
	
	scale = 10 / max_val
	threshold = np.percentile(kdistances, 95) * scale
	
	for distance in kdistances:
		val = distance * scale
		outlier_vals.append(val)
		if val > threshold:
			outlier_class.append('outlier')
		else:
			outlier_class.append('not outlier')

	return outlier_vals, outlier_class

"""
Uses read_data.py to gather points for outlier detection
transforms list of points to array
"""
def readData():
	points = np.array(read_data.main())
	return points

"""
Uses matplotlib to generate a scatter plot of data
outliers are shown on a scale from red to blue
red being a point deviating the most from the others
"""
def plot_points(vor, points, k_distances, k, axis_labels):
	voronoi_plot_2d(vor, show_points=False, show_vertices=False, line_width=0.5)
	plot = plt.scatter([item[0] for item in points],
 		[item[1] for item in points],
 		c=k_distances,
		s=3,
 		cmap="coolwarm")
	plt.xlabel(axis_labels[0])
	plt.ylabel(axis_labels[1])

	cbar = plt.colorbar(plot)
	cbar.set_label('Voronoi {k}-distance'.format(k=k), rotation=270, labelpad=15)
	

#starting point
if __name__ == "__main__":
	global regions
	global neighbor_ids

	neighbor_ids = []
	points, axis_labels, k = read_data.main()
	
	vor = Voronoi(points, furthest_site = False)

	vertices = vor.vertices
	ridge_vertices = vor.regions
	regions = vor.regions
	point_region = vor.point_region

	# Generate the k-distance for each point
	k_distances = [kDistance(i, k) for i in range(len(points))]
	outlier_values,outlier_class = getOutlierValues(k_distances)
	perimeters = [region_perimeter(i) for i in range(len(points))]
		
	df = pd.DataFrame(data=points, columns=axis_labels)
	df = df.assign(perimeter=perimeters)
	df = df.assign(outliervalue=outlier_values)
	df = df.assign(outlierclass=outlier_class)
	df = df.assign(neighbors=neighbor_ids)
	df = df.sort_values(by=['outliervalue'],ascending=False)

	df.to_csv("./result.csv")

	print(df)

	plot_points(vor,points,k_distances,k,axis_labels)
	plt.show()
