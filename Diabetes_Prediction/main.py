import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

if (__name__ == "__main__"):
    # Import dataset into pandas
    df = pd.read_csv('diabetes.csv')
    print(df.head())
    # Create and plot histogram for .csv parameters
    df.hist() ; plt.show() ; plt.pause(1) ; plt.close()
    # Create a subplot 3x3 
    plt.subplots(3,3,figsize=(15,15))
    # Plot a distribution density plot for each variable
    for idx,col in enumerate(df.columns):
        ax = plt.subplot(3,3,idx+1)
        ax.yaxis.set_ticklabels([])
        sns.distplot(df.loc[df.Outcome == 0][col], hist=False, axlabel= False, \
            kde_kws={'linestyle':'-','color':'black', 'label':"No Diabetes"})
        sns.distplot(df.loc[df.Outcome == 1][col], hist=False, axlabel= False,\
             kde_kws={'linestyle':'--','color':'black', 'label':"Diabetes"})
        ax.set_title(col)
        # Hide the 9th subplot (bottom right) since there are only 8 plots
    plt.subplot(3,3,9).set_visible(False)
    plt.show(); plt.pause(1) ; plt.close()
    # Check if there are missing values in dataset because  there are misdirections
    # on histograms and distribution plots.
    print(df.isnull().any())
    # Further investigation for those values 
    print(df.describe()) 
    # Handle zero values on crucial variables
    # Replace 0 values with NaN on <Glucose>,<BloodPressure>,<SkinThickness>
    # <Insulin> and <BMI> variables
    df['Glucose'] = df['Glucose'].replace(0,np.nan)
    df['BloodPressure'] = df['BloodPressure'].replace(0,np.nan)
    df['SkinThickness'] = df['SkinThickness'].replace(0,np.nan)
    df['Insulin'] = df['Insulin'].replace(0,np.nan)
    df['BMI'] = df['BMI'].replace(0,np.nan)
    # we will replace those NaN values with mean of non-missing values
    df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
    df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
    df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
    df['BMI'] = df['BMI'].fillna(df['BMI'].mean())
    # create a scaled dataset for our values
    df_scaled = preprocessing.scale(df)
    # the object that created by preprocessing is no longer <DataFrame> so we 
    # need to convert it back
    df_scaled = pd.DataFrame(df_scaled,columns = df.columns)
    # we need to keep the same <Outcome> value without scaling
    df_scaled['Outcome'] = df['Outcome']
    df = df_scaled
    print(df.describe().loc[['mean','std','max'],].round(2).abs())
    # Separate dataset into input features(X) and target variable(y)
    X = df.loc[:, df.columns != 'Outcome']
    y = df.loc[:,'Outcome']
    # Split data into train and testing data by using 20% as testing data
    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2)
    # Split training data into training and validation set by 20% rule
    X_train , X_val , y_train , y_val = train_test_split(X,y,test_size = 0.2)
    # Creation of neural network
    model = Sequential()
    # Add the first hidden layer
    model.add( Dense(32,activation = 'relu',input_dim = 8) )
    # Add the second hidden layer
    model.add( Dense(16,activation = 'relu'))
    # Add the output layer 
    model.add( Dense(1,activation = 'sigmoid'))
    # Compile the model by defining Optimizer, Loss function and Metrics
    model.compile(optimizer = 'adam' , \
        loss = 'binary_crossentropy' , metrics=['accuracy'] )
    # Train the model 
    model.fit(X_train,y_train,epochs= 200,verbose = 0)
    print("   ")
    # Evaluation of model's accuracy for training and testing data  
    scores_train = model.evaluate(X_train, y_train)
    print("Training Accuracy: %.2f%%\n" % (scores_train[1]*100))
    scores_test = model.evaluate(X_test, y_test)
    print("Testing Accuracy: %.2f%%\n" % (scores_test[1]*100))
    # Evaluation through confusion matrix 
    y_test_pred = model.predict_classes(X_test)
    # Creation of confusion matrix
    c_matrix = confusion_matrix(y_test,y_test_pred)
    # Plotting of confusion matrix 
    ax = sns.heatmap(c_matrix, annot = True, xticklabels=['No Diabetes','Diabetes'], \
        yticklabels=['No Diabetes','Diabetes'],cbar=False, cmap='Blues')
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    plt.show(); plt.pause(1); plt.close()
    # Evaluation through ROC curve
    y_test_pred_probs = model.predict(X_test)
    FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
    plt.plot(FPR, TPR ,linewidth=3, color = 'red')
    plt.plot([0,1],[0,1],'--', color='black') #diagonal line
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show() ; plt.pause(1); plt.close()



