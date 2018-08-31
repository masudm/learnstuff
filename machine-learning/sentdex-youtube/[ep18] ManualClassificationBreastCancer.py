from math import sqrt
import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbour(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is too low for amount of features.')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    #get k top groups in sorted distances (so closest groups)
    votes = [i[1] for i in sorted(distances)[:k]]

    #the most common group
    #it returns array of tuple (class, how common) so use [0][0]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

#load in dataset
df = pd.read_csv("ep14data.txt")
#replace the unknown values with an outlier
df.replace("?", -99999, inplace=True)
#drop the id column as it does not affect classification (only in bad way)
df.drop(["id"], 1, inplace=True)

#convert everything to a float
full_data = df.astype(float).values.tolist()

#shuffle the data
random.shuffle(full_data)

#20% of data is test
test_size = 0.2

#empty dict to store data
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}

#get 80%
train_data = full_data[:-int(test_size*len(full_data))]
#get last 20%
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    #the last value (which is the class in this dataset - 2 (benign) or 4 (malignant))
    #and append data to that (i.e. the empty dict)
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    #the last value (which is the class in this dataset - 2 (benign) or 4 (malignant))
    #and append data to that (i.e. the empty dict)
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

#for each class (2 and 4)
for group in test_set:
    #for each data point...
    for data in test_set[group]:
        #get the vote (which class it thinks) by training and prediciting against test data
        vote = k_nearest_neighbour(train_set, data, k=5)

        #if it's correct...
        if group == vote:
            correct += 1
        
        #total amount of times it tried to predict
        total += 1

accuracy = correct/total
print("Accuracy", accuracy)