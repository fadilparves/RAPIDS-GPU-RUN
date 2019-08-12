import dask.dataframe as pd
import time
import numpy as np
from math import cos, sin, asin, sqrt, pi
import os

start = time.time()
np.random.seed(12)
data_length = 1000000

df = pd.read_csv("data.csv")

def haversine_distance_kernel(row):
       
        x_1 = pi/180 * row['lat1']
        y_1 = pi/180 * row['lon1']
        x_2 = pi/180 * row['lat2']
        y_2 = pi/180 * row['lon2']
        
        dlon = y_2 - y_1
        dlat = x_2 - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_2) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        return c * r

df['out'] = df.apply(haversine_distance_kernel, meta=(None, 'float64'), axis=1)
df.to_csv("f-*.csv")
end = time.time()
diff = end - start
print("Time taken in " + str(diff))

