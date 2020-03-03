import numpy as np

def preprocessing(dataset,nyc_limits,low_fare = 2.5,high_fare = 100.0,passenger_lim = 8):

    dataset = dataset.dropna()
    # remove outliers from fare amount
    dataset = dataset[ (dataset['fare_amount'] > low_fare) & 
                    (dataset['fare_amount'] <= high_fare) ]

    # remove outliers from passenger_count 
    dataset = dataset[ (dataset['passenger_count'] < passenger_lim) ]
    dataset.loc[ dataset['passenger_count'] == 0 , 'passenger_count'] = 1

    # remove longitude and latitude outliers

    for longi in ['pickup_longitude', 'dropoff_longitude']:
        dataset = dataset[(dataset[longi] > nyc_limits[0]) & 
                        (dataset[longi] < nyc_limits[1])]
    for lati in ['pickup_latitude', 'dropoff_latitude']:
        dataset = dataset[(dataset[lati] > nyc_limits[2]) & 
                        (dataset[lati] <nyc_limits[3])] 
    
    return dataset

def feature_engineering(df,airports):

    # geolocation features

    def haversine_dist(f1,l1,f2,l2):
        R = 6371 # Earth's radius 
        # Convert to radians from degrees
        f1_r = f1 * np.pi / 180 ; f2_r = f2 * np.pi / 180
        l1_r = l1 * np.pi / 180 ; l2_r = l2 * np.pi / 180 

        haversine_f = ( 1-np.cos(f2_r - f1_r) )/2 
        haversine_l = ( 1-np.cos(l2_r - l1_r) )/2
        haversine_th =  haversine_f + np.cos(f1_r)*np.cos(f2_r)*haversine_l
        d = 2 * R * np.arcsin(np.sqrt(haversine_th))
        return d

    # Convert <pickup_datetime> into more specific feature
    df['year'] = (df['pickup_datetime'].dt.year).astype('uint16')
    df['month'] = (df['pickup_datetime'].dt.month).astype('uint8')
    df['day'] = (df['pickup_datetime'].dt.day).astype('uint8')
    df['day_of_week'] = (df['pickup_datetime'].dt.dayofweek).astype('uint8')
    df['hour'] = (df['pickup_datetime'].dt.hour).astype('uint8')

    df = df.drop(['pickup_datetime'],axis = 'columns')

    # Compute Haversine Distance
    df['haversine_distance'] = haversine_dist(df['pickup_latitude'],df['pickup_longitude'],
                                     df['dropoff_latitude'],df['dropoff_longitude'])

    df = df[ df['haversine_distance'] > 0.0]

    for k in airports:
        df['haversine_pickup_distance_'+k] = haversine_dist(df['pickup_latitude'],
                                        df['pickup_longitude'],airports[k][1],
                                        airports[k][0])
        df['haversine_dropoff_distance_'+k] = haversine_dist(df['dropoff_latitude'],
                                        df['dropoff_longitude'],airports[k][1],
                                        airports[k][0])

    return df
   
def predict_random(df_prescaled, X_test, model):
        sample = X_test.sample(n=1,random_state = np.random.randint(low = 0, high = 300_000))
        idx = sample.index[0]
        actual_fare = df_prescaled.loc[idx,'fare_amount']
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                        'Saturday', 'Sunday']
        day_of_week = day_names[df_prescaled.loc[idx,'day_of_week']]
        hour = df_prescaled.loc[idx,'hour']
        predicted_fare = model.predict(sample)[0][0]
        rmse = np.sqrt(np.square(predicted_fare-actual_fare))
        
        print("Trip Details: {}, {}:00".format(day_of_week, hour))
        print("Actual fare: ${:0.2f}".format(actual_fare))
        print("Predicted fare: ${:0.2f}".format(predicted_fare))
        print("RMSE: ${:0.2f}".format(rmse))
        print(" ")
        

def predict_airport(df_prescaled,X_test,model):
    import pandas as pd

    samples = df_prescaled.loc[df_prescaled['fare_amount'] == 52.00]
    idx = samples.index.tolist()
    t_idx = X_test.index.tolist()
    idx_ideal = None
    for i in idx: 
        if i in t_idx: 
            idx_ideal = i
            break
    actual_fare = df_prescaled.loc[idx_ideal,'fare_amount']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                        'Saturday', 'Sunday']
    day_of_week = day_names[df_prescaled.loc[idx_ideal,'day_of_week']]
    hour = df_prescaled.loc[idx_ideal,'hour']
    airport_sample = (X_test.iloc[idx_ideal]).to_frame().T 
    predicted_fare = model.predict(airport_sample)[0][0]    
    rmse = np.sqrt(np.square(predicted_fare-actual_fare))

    print("Trip Details: {}, {}:00".format(day_of_week, hour))
    print("Actual fare: ${:0.2f}".format(actual_fare))
    print("Predicted fare: ${:0.2f}".format(predicted_fare))
    print("RMSE: ${:0.2f}".format(rmse))
    print(" ")
    