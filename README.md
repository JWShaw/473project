# CPSC 473 Term Project: Voronoi Diagram and Voronoi k-Distance

This project generates the Voronoi diagram in Euclidean 2-space for an arbitrary set of points and uses this partition to identify datapoints with outlier-like characteristics.

![](Figure_1.pdf)

Built by Jonathan Shaw, Kai Lan, and Sofia Jones for CPSC 473: Data Mining.

## Setting Up and Running

Dependencies:

* `pandas`
* `numpy`
* `matplotlib`
* `scipy`
* `sklearn`

### Voronoi Diagram Generation

To test our home-grown implementation of the Voronoi diagram generation algorithm, run `python diagram_generation.py` from the project root directory.
Since this is simply a proof-of-concept, no input need be provided: it will generate a set of random datapoints and plot the corresponding Voronoi diagram.
Note that our implementation is flawed, so the diagram is occasionally missing edges for some inputs.

### Voronoi k-Distance Outlier Detection

The main component of the project is the efficient density-based outlier detection algorithm we implemented.  To try it, run `python voronoi-k-distance.py`
followed by the dataset you wish to detect outliers in.  For example, for `ahmedabad.csv`, one would run `python voronoi-k-distance.py ./datasets/amhedabad.csv`.

Optionally, the desired value of `k` can be provided via a second argument.  For example, to find outliers for clusters.csv via voronoi 2-distance, one would run
`python voronoi-k-distance.py ./datasets/clusters.csv 2`.  If no second argument is provided, the 3-distance is used by default.

One can (and should) take the time to play with different datasets and values of `k`.  For a beginner, `clusters.csv` and `ahmedabad.csv` are good datasets to start with.

An output file, `output.csv`, is created in the project root directory whenever the program is run.  The output file format is described in the formal report.