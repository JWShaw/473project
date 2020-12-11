import pandas as pd
import numpy as np
import sys

def main():
	try:
		filepath = sys.argv[1]
		df = pd.read_csv(filepath)
		df = df.dropna()

		points = df.values.tolist()
		axis_labels = list(df)

		k = 3
		if len(sys.argv) == 3:
			k = int(sys.argv[2])

		return points, axis_labels, k
	except:
		sys.exit("Please input file path!")

main()
