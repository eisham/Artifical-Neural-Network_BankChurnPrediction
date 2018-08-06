# Part-2 --> Building the architecture of ANN

# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# Initialising the ANN as successive layers 
# Here we create an object of Sequential class which defines ANN as a sequence of layers
classifier = Sequential()


# Adding Input layer and First hidden layer
# Add method is used to add layers to the ANN. For the first hidden layer we need to mention the imput_dim of the input layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

# Adding second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


# Compiling ANN
#adam is the stochastic gradient descent algorithm used to optimize the weights each time an epochs is executed
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy']) 


# Fitting the ANN to the Training set
# Randomly assigned batch size and epochs
classifier.fit(X_train, y_train, batch_size=10, epochs=100) 