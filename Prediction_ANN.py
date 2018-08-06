# Part-3 --> Making Prediction

y_pred = classifier.predict(X_test)

# This step is to convert the probability into actual prediction -> 1 or 0
y_pred = (y_pred>0.5) 


# Predicting a single new observation
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5) 


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

# Confusion matrix is used to measure accuracy of the prediction on test set
cm = confusion_matrix(y_test, y_pred) 