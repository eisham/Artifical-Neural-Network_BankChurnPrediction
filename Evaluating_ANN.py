# Part-4 -->Evaluating the ANN

#Its a keras wrapper which will wrap the scikit learn function into keras model
from keras.wrappers.scikit_learn import KerasClassifier 
# Its a k fold cross validation function from scikit learn library
from sklearn.model_selection import cross_val_score 


# Function for building the ANN
# Dropout Regularization to reduce overfitting
def build_classifier(): 
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dropout(p= 0.1)) # start with 0.1 and if overfitting isnt reduced; increase stepwise -> 0.2, 0.3, don't go beyong 0.5; as it will lead to underfitting
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dropout(p= 0.1))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


# This classifier object is of keras wrapper and it takes on to train/fit the model with building neural network,batch,epochs
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)


# accuracies is an object of k fold cross validation function 
# parameter-cv tells the number of k folds validations we want to do; advisable to use cv=10
# n_jobs= -1 allows parallel computation utilizing all CPUs/GPUs across the k fold validations performed
accuracies = cross_val_score(estimator= classifier, X= X_train, y=y_train, cv= 10, n_jobs= -1)  


# accuracies is the array of 10 accuracies calculated for 10 fold trainings
# computing mean to get the average accuracy
# computing variance to see their deviation from the mean
mean = accuracies.mean()
variance = accuracies.std()

# Expected is to get low bias (high accuracy) and low variance (less deviation of accuracies from each other)