# Part-5 --> Tuning ANN
# Tuning the ANN for improvization in accuracy

# Hyperparameters are the static parameters passed to the model for training like batch size, epochs, algorithm

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer): 
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier)


# Below is a dictionary of hyperparameters for experimentation
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}


grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)


grid_search = grid_search.fit(X_train, y_train)


# To know which hyperparameters worked the best
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_