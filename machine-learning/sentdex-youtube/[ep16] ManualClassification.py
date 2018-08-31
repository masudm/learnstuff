from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

#two classes. features: k,r and their data
dataset = {'k': [[1,2], [2,3], [3,1]], 'r': [[6,5], [7,7], [8,6]]}

#predict this one
new_features = [5,7]

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

result = k_nearest_neighbour(dataset, new_features, k=3)
print(result)

#graph the dataset
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)

#plot the new (one to predict)
plt.scatter(new_features[0], new_features[1], s=50, color=result)

plt.show()
