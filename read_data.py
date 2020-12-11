import pandas as pd
import numpy as np

def create_df(filepath):
	df = pd.read_csv(filepath)
	return df

def drop_nulls(df, col):
	df = df[df[col].notna()]
	return df

def getRows(df, value):
	rows = df.loc[df['AQI_Bucket'] == value]
	return rows
	
def main():
	full_df = create_df("archive/city_day.csv")
	df_subset = full_df[:3000]

	poor_aqi = getRows(df_subset, 'Very Poor')
	mod_aqi = getRows(df_subset, 'Moderate')
	good_aqi = getRows(df_subset, 'Good')

	frames = [poor_aqi, mod_aqi, good_aqi]
	df = pd.concat(frames)

	df = drop_nulls(df,'AQI')
	df = drop_nulls(df, 'PM2.5')	
	AQI = df['AQI'].tolist()
	PM = df['PM2.5'].tolist()
	points = list(zip(AQI, PM))
	
	return points
main()
'''
	

print(df.columns)
'''
