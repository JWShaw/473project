import math
import random
import matplotlib.pyplot as plt

def randData(num_points):
	x = [random.triangular() for i in range(num_points)]
	y = [random.triangular() for i in range(num_points)]
	colors = [random.randint(1,10) for i in range(num_points)]
	areas = [random.randint(50,100) for i in range(num_points)]
	draw_plot(x, y, areas, colors)

def draw_plot(x, y, areas, colors):
	plt.figure()
	plt.scatter(x, y, s=areas, c=colors, alpha=0.85)
	plt.axis([0.0,1.0,0.0,1.0])
	plt.show()

randData(100);

