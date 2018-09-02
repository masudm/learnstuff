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
        #we do not need as fine steps with b
        b_range_multiple = 5
        b_multiple = 5

        #
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimised = False
            while not optimised:
                #we want min w and max b
                for b in np.arange(-1 * self.max_feature_value * b_range_multiple, self.max_feature_value*b_range_multiple, step*b_multiple):
                    for transformation in transforms:
                        #transform w with each transformation
                        w_t = w*transformation
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t, xi) + b) >= 1:
                                    found_option = False

                    if found_option:
                        #linalg is magnitude of vector
                        opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimised = True
                    print('Optimised a step.')
                else:
                    w = w - step

            #get smallest magnitude by sorting
            norms = sorted([n for n in opt_dict])

            #get the w and b from the smallest magnitude (which is key remember)
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            
            #use the smaller step value
            latest_optimum = opt_choice[0][0]+step*2

    #predict new data
    def predict(self, features):
        #the basic algorithm
        #run x.w + b and find out if + or -
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        if classification != 0 and self.visualisation:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])

        return classification

    def visualise(self):
        [[self.ax.scatter(x[0], x[1], s=100, c=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        #v = x.w + b
        # +ve, v = 1    -ve, v =-1  decision boundary, v = 0
        def hyperplane(x, w, b ,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        #+ve hyperplane
        #these are y points
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        #-ve hyperplane
        #these are y points
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        #decision boundary hyperplane
        #these are y points
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


#create a demo dataset to train with
#two classes: -1, 1
data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8]]), 1: np.array([[5, 1], [6, -1], [7, 3]])}

svm = Support_Vector_Machine()
svm.fit(data = data_dict)

predict_us = [
    [0 ,10], 
    [1, 3],
    [3, 4],
    [3, 5],
    [5, 5],
    [5, 6],
    [6, -5],
    [5, 8]
]

for p in predict_us:
    svm.predict(p)

svm.visualise()

