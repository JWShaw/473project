import pandas as pd
import numpy as np
import sys


def cluster(df,k):
	k_clusters = []
	first_col_name = df.columns[0]
	first_col = df.iloc[:,0]
	clusters = first_col.unique()

	new_df = []
	for i in range(k):
		row = df.loc[df[first_col_name] == clusters[i]]
		new_df.append(row)

	df = pd.concat(new_df)
	points = list(zip(df.iloc[:,1].tolist(),df.iloc[:,2].tolist()))
	
	return points
		
		
		
def main():
	try:
		filepath = sys.argv[1]
		df = pd.read_csv(filepath)
		df = df.dropna()

		num_cols = len(df.columns)
		k = 3

		if len(sys.argv) == 3:
			k = int(sys.argv[2])

		if num_cols > 2:
			df = df[:3000]
			points = cluster(df,k)
			df = df.drop(columns = df.columns[0])
			axis_labels = list(df)	
		else:	
			points = df.values.tolist()
			axis_labels = list(df)

		return points, axis_labels, k
	except:
		sys.exit("Please input file path!")

main()
