import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

#load in dataset: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
df = pd.read_csv("ep14data.txt")

#get rid of the ? and replace with another value that is treated as an outlier
df.replace("?", -99999, inplace=True)

#drop the id column as it is not related to dataset
df.drop(["id"], 1, inplace=True)

#define x (features) and y (labels)
X = np.array(df.drop(["class"], 1))
y = np.array(df["class"])

#shuffle and split data into train and test data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#setup classifier
clf = neighbors.KNeighborsClassifier()

#fit the classifier to the train data
clf.fit(X_train, y_train)

#create an accuracy var
accuracy = clf.score(X_test, y_test)
print("accuracy: " + str(accuracy))

#create an example dataset to predict
example_measures = np.array([[4,2,1,1,1,2,3,4,1]])
#resize the array to get ride of error
example_measures = example_measures.reshape(len(example_measures), -1)

#predict using example measure
prediction = clf.predict(example_measures)
print(prediction)