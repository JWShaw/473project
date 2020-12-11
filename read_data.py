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

		return points, axis_labels
	except:
		sys.exit("Please input file path!")

main()
