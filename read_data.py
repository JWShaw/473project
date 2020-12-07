import pandas as pd
import numpy as np

def create_df(filepath):
	df = pd.read_csv(filepath)
	return df

def drop_nulls(df, col):
	df = df[df[col].notna()]
	return df

def main():
	df = create_df("archive/city_day.csv")
	df = drop_nulls(df,'PM2.5')
	df = drop_nulls(df, 'AQI')	
	AQI = df['AQI'].tolist()
	PM = df['PM2.5'].tolist()
	points = list(zip(AQI, PM))
	return points
main()
'''
	

print(df.columns)
'''
