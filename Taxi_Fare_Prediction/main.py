import time 

def main() :
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from visualization import plot_lat_long_on_map,plot_features_histograms
    from utilization import preprocessing,feature_engineering
    from utilization import predict_random,predict_airport
    from sklearn.preprocessing import scale
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.metrics import mean_squared_error

    # Load .csv file with appropriate types to reduce time 
    types = {'fare_amount': 'float32',
         'pickup_longitude': 'float32',
         'pickup_latitude': 'float32',
         'dropoff_longitude': 'float32',
         'dropoff_latitude': 'float32',
         'passenger_count': 'uint8'}

    # range of NYC longitude
    nyc_min_longitude,nyc_max_longitude = -74.05 , -73.75
    # range of NYC latitude
    nyc_min_latitude,nyc_max_latitude = 40.63 , 40.85
    limits = (nyc_min_longitude,nyc_max_longitude,nyc_min_latitude,nyc_max_latitude )
    # Major NYC airports longitudes and latitudes 
    airports = {'JFK Airport': (-73.78, 40.643),
            'Laguardia Airport': (-73.87, 40.77),
           	'Newark_Airport' : (-74.18, 40.69)}
    
    df = pd.read_csv('NYC_taxi.csv',nrows = 1_000_000,parse_dates = ['pickup_datetime'],
                    infer_datetime_format=True,dtype=types)
    
    # 'key' feature is similat to 'pickup_datetime', so we will remove it 
    df = df.drop(['key'],axis = 'columns')

    # Data Preproccesing
    df = preprocessing(df,limits)

    # Plot real pickup and drop off points with some key location of NYC
    
    #plot_lat_long_on_map(df,BB = limits,points = 'Pickup')
    #plot_lat_long_on_map(df,BB = limits,points = 'Dropoff')
    
    # Day of Week histogram
    # plot_features_histograms(df,"day_of_week")
    # Pickup Hour histogram
    # plot_features_histograms(df,"pickup_hour")

    # Feature Enginnering
    df = feature_engineering(df,airports)

    #df.plot.scatter('fare_amount','haversine_distance')
    #plt.xlabel("Fare Amount ($)")
    #plt.ylabel("Distance (in km)")
    #plt.title("Fare Amount - Distance Correlation")
    #plt.show()
   
    # Parameter scaling
    df_prescaled = df.copy()
    df_scaled = df.drop(['fare_amount'],axis = 'columns')
    df_scaled = scale(df_scaled)
    cols = list(df.columns) ; cols.remove('fare_amount')
    df_scaled = pd.DataFrame(df_scaled,columns = cols,index = df.index)
    df_scaled = pd.concat([df_scaled,df['fare_amount']],axis = 'columns')
    df = df_scaled.copy()
    
    #print(df.head())

    # Deep Neural Network Model

    # Split the df into training features (x) and target we are going to predict y
    X = df.loc[:,df.columns != 'fare_amount']
    Y = df.loc[:,'fare_amount']

    # Split into training set (80%) and test set (20%)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2)

    # X_test --> DataFrame
    
    nodes = 128
    model = Sequential()
    model.add(Dense(nodes, activation= 'relu', input_dim=X_train.shape[1]))
    model.add(Dense(nodes//2, activation= 'relu'))
    model.add(Dense(nodes//4, activation= 'relu'))
    model.add(Dense(nodes//8, activation= 'relu'))
    model.add(Dense(nodes//16, activation= 'relu'))
    model.add(Dense(1))
    
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.fit(X_train, Y_train, epochs=1)

    predict_random(df_prescaled, X_test, model)
    predict_random(df_prescaled, X_test, model)
    predict_random(df_prescaled, X_test, model)
    predict_airport(df_prescaled, X_test, model)

    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(Y_train, train_pred))

    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(Y_test, test_pred))

    print("")
    print("Train RMSE: {:0.2f}".format(train_rmse))
    print("Test RMSE: {:0.2f}".format(test_rmse))
    print("")

if (__name__ == '__main__'):
    start=time.time()
    main()
    print("%d" %((time.time()-start)/60) , "minutes")
