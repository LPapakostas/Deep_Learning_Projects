import matplotlib.pyplot as plt
import numpy as np

nyc_landmarks = {'JFK Airport': (-73.78, 40.643),
             'Laguardia Airport': (-73.87, 40.77),
             'Midtown': (-73.98, 40.76),
             'Lower Manhattan': (-74.00, 40.72),
             'Upper Manhattan': (-73.94, 40.82),
             'Brooklyn': (-73.95, 40.66),
             'Queens': (-73.77, 40.74)}

def plot_lat_long_on_map(df, BB, landmarks = nyc_landmarks , points='Pickup'):

    nyc_map = plt.imread('nyc_map.png')
    plt.figure(figsize = (12,12)) # set figure size
    plt.imshow(nyc_map , extent = BB)

    if points == 'Pickup':
        plt.plot(list(df.pickup_longitude), list(df.pickup_latitude),
                '.', markersize=1)
    else:
        plt.plot(list(df.dropoff_longitude), list(df.dropoff_latitude),
                '.', markersize=1,color ='k')
    
    for landmark in landmarks:
        plt.plot(landmarks[landmark][0], landmarks[landmark][1],
            '*', markersize=15, alpha=1, color='r')
        plt.annotate(landmark, (landmarks[landmark][0]+0.005,
            landmarks[landmark][1]+0.005), color='r',
            backgroundcolor='w')

    plt.title("{} Locations in NYC Illustrated".format(points))
    plt.grid(None)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    
    plt.show()

def plot_features_histograms(dataset,desc):

    if (desc == "day_of_week"):
        dataset['day_of_week'].plot.hist(bins = np.arange(8) - 0.5,ec ='black',ylim = (115_000,150_000))
        plt.xlabel("Day Of Week (Monday = 0 , Sunday = 6)")
        plt.ylabel("Number Of Passengers")
        plt.title("Day Of Week Histogram")
        plt.show()
    elif (desc == "pickup_hour"):
        dataset['hour'].plot.hist(bins = np.arange(25) - 0.5,ec ='black')
        plt.xlabel("Hour")
        plt.ylabel("Number Of Passengers")
        plt.title("Pickup Hour Histogram")
        plt.show()
    else:
        raise Exception("False Description at <desc> variable")

    