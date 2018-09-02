import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use("ggplot")

#create a class for the SVM
class Support_Vector_Machine:
    #when creating the object
    def __init__(self, visualisation=True):
        #visualise the dataset?
        self.visualisation = visualisation
        #colours for each of the classes
        self.colors = {1: 'r', -1: 'b'}

        #if visualising
        if (self.visualisation):
            #create a figure/window
            self.fig = plt.figure()
            #plot a 1x1 grid
            self.ax = self.fig.add_subplot(1, 1, 1)

    #train the data
    def fit(self, data):
        #make sure entire class can access data
        self.data = data

        # a dictionary: key is magnitude of w and key is [w, b]
        opt_dict = {}

        #apply to the vector each 'step'
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        #create a list to store all the features
        all_data = []

        #iterate through to append the features
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        #get the max and min of the features
        #you can use this to create step sizes
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        #get it out of memory
        all_data = None

        #smaller stops as it overfits
        step_sizes = [self.max_feature_value * 0.1, self.max_feature_value * 0.01, self.max_feature_value * 0.001]

        #the step range of b
        b_range_multiple = 5

        #
        b_multiple = 5

        #
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimised = False
            while not optimised:
                pass

    #predict new data
    def predict(self, features):
        #the basic algorithm
        #run x.w + b and find out if + or -
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        return classification



#create a demo dataset to train with
#two classes: -1, 1
data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8]]), 1: np.array([[5, 1], [6, -1], [7, 3]])}

