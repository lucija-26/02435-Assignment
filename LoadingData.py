import numpy as np

# loading occupancy in room 1
occupancy_room1 = np.loadtxt('OccupancyRoom1.csv', delimiter=',', skiprows=1)

# loading occupancy in room 2
occupancy_room2 = np.loadtxt('OccupancyRoom2.csv', delimiter=',', skiprows=1)

# loading price data
price_data = np.loadtxt('PriceData.csv', delimiter=',', skiprows=1)

