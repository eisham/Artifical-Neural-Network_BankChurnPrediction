# Artificial Neural Network

# Part-1 --> Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Remember to impute and/or encode the datasets before slitting into train and test set
# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) # labelencoder converts categorical to numerical
#labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_1.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray() # converts the numerical values to dummy variables as values doesn't have any order 
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part-2 --> Making ANN

# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN as successive layers 
# Here we create an object of Sequential class which defines ANN as a sequence of layers
classifier = Sequential()

# Adding Input layer and First hidden layer
# Add method is used add layers to the ANN, For the first hidden layer we need to mention the imput_dim which is the input layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

# Adding second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

# Compiling ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy']) #adam is the stoicistic gradient descend algorithm used to optimize the weights each time an epochs is executed

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100) # Randomly assigned batch size and epochs

# Part-3 --> Making the prediction and evaluating the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)  # This step is to convert the probability into actual predcition of 1 or 0

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 
 
# Part-4 -->Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier #Its a keras wrapper which will wrap the scikit learn function into keras model
from sklearn.model_selection import cross_val_score # Its a k fold cross validation function from scikit learn library

def build_classifier(): # classifier object inside the function is local to the function itself
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# This classifier object is of keras wrapper and it takes on to train/fit the model with building neural network,batch,epochs
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)

# accuracies object is object of k fold cross validation function which takes the neural network built, training arrays
# cv tells the number of k folds we want to do (advisable to use 10 and n_jobs is to allow parallel computation using all CPUs as the model is trained 10 times with batch and epochs on each of them running parallely)
accuracies = cross_val_score(estimator= classifier, X= X_train, y=y_train, cv= 10, n_jobs= -1)  

# accuracies is the array of 10 accuracies calculated for 10 fold tranings
# computing mean to get the average accuracy and variance is calculated to see how much the different accuracies vary from the mean
# Expected is to get low bias (high accuracy) and low variance (less deviation of accuracies from each other)
mean = accuracies.mean()
variance = accuracies.std()


# 








