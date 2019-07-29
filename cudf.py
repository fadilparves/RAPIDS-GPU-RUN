from math import cos, sin, asin, sqrt, pi
import time
import cudf as dd
import numpy as np
from numba import cuda

start = time.time()
np.random.seed(12)
data_length = 1000

df = dd.DataFrame()
df['lat1'] = np.random.normal(10, 1, data_length)
df['lon1'] = np.random.normal(10, 1, data_length)
df['lat2'] = np.random.normal(10, 1, data_length)
df['lon2'] = np.random.normal(10, 1, data_length)

def haversine_distance_kernel(lat1, lon1, lat2, lon2, out):
    """Haversine distance formula taken from Michael Dunn's StackOverflow post:
    https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    for i, (x_1, y_1, x_2, y_2) in enumerate(zip(lat1, lon1, lat2, lon2)):
        print('thread_id:', cuda.threadIdx.x, 'bid:', cuda.blockIdx.x,
              'array size:', lat1.size, 'block threads:', cuda.blockDim.x)

        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_2 = pi/180 * x_2
        y_2 = pi/180 * y_2
        
        dlon = y_2 - y_1
        dlat = x_2 - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_2) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        out[i] = c * r

df = df.apply_rows(haversine_distance_kernel,
                   incols=['lat1', 'lon1', 'lat2', 'lon2'],
                   outcols=dict(out=np.float64),
                   kwargs=dict())

print(df)
end = time.time()
diff = end - start
print("Time Taken is " + str(diff))