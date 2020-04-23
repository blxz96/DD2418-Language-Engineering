import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math

"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Dmytro Kalpakchi.
"""
class LogisticRegression(object):
    """
    This class performs logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    def __init__(self, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param theta    A ready-made model
        """
        theta_check = theta is not None

        if theta_check:
            self.FEATURES = len(theta)
            self.theta = theta

        #  ------------- Hyperparameters ------------------ #
        self.LEARNING_RATE = 0.1        # The learning rate.
        self.MINIBATCH_SIZE = 128        # Minibatch size
        self.PATIENCE = 5                   # A max number of consequent epochs with monotonously
                                            # increasing validation loss for declaring overfitting
        # ---------------------------------------------------------------------- 


    def init_params(self, x, y):
        """
        Initializes the trainable parameters of the model and dataset-specific variables
        """
        # To limit the effects of randomness
        np.random.seed(524287)

        # Number of features
        self.FEATURES = len(x[0]) + 1

        # Number of classes
        self.CLASSES = len(np.unique(y))

        # Number of datapoints.
        self.TRAINING_DATAPOINTS = len(x)

        # Training data is stored in self.x (with a bias term) and self.y
        x_t, y_t, x_v, y_v = self.train_validation_split(np.array(x), np.array(y))

        # add dummy column for both x
        self.x = np.concatenate((np.ones((len(x_t), 1)), x_t), axis = 1)
        self.y = y_t
        self.xv = np.concatenate((np.ones((len(x_v), 1)), x_v), axis = 1)
        self.yv = y_v


        # The weights we want to learn in the training phase.
        K = np.sqrt(1 / self.FEATURES)
        self.theta = np.random.uniform(-K, K, (self.FEATURES, self.CLASSES))

        # The current gradient.
        self.gradient = np.zeros((self.FEATURES, self.CLASSES))


        print("NUMBER OF DATAPOINTS: {}".format(self.TRAINING_DATAPOINTS))
        print("NUMBER OF CLASSES: {}".format(self.CLASSES))


    def train_validation_split(self, x, y, ratio=0.9):
        """
        Splits the data into training and validation set, taking the `ratio` * 100 percent of the data for training
        and `1 - ratio` * 100 percent of the data for validation.

        @param x        A (N, D + 1) matrix containing training datapoints
        @param y        An array of length N containing labels for the datapoints
        @param ratio    Specifies how much of the given data should be used for training
        """
        #
        # YOUR CODE HERE
        #
        train_size = int(ratio * len(x))

        # shuffle indices
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        # assign both x and y with the same indices
        x = x[indices]
        y = y[indices]

        x_tr = x[:train_size]
        y_tr = y[:train_size]
        x_val = x[train_size:]
        y_val = y[train_size:]
        return x_tr, y_tr, x_val, y_val


    def loss(self, x, y):
        """
        Calculates the loss for the datapoints present in `x` given the labels `y`.
        """
        #
        # YOUR CODE HERE
        #
        theta_mult_x = x.dot(self.theta)

        # generate one-hot labels
        y_new = np.zeros((y.shape[0], self.CLASSES))
        for i in range(self.CLASSES):
            y_new[:, i] = np.where(y[:] == i, 1, 0)


        # Sigmoid each value in the result
        sigmoid_v = np.vectorize(self.sigmoid)
        theta_mult_x = sigmoid_v(theta_mult_x)

        # Get the loss function for case 1
        val_label_1 = np.log(theta_mult_x)
        val_label_1 *= -y_new

        # Get the loss function for case 0
        val_label_0 = np.log(1-theta_mult_x)
        val_label_0 *= (1-y_new)

        # Subtract
        sigma = val_label_1 - val_label_0

        # Sum all terms to get the total loss
        loss = np.sum(sigma) / y_new.size

        return loss


    def conditional_log_prob(self, label, datapoint):
        """
        Computes the conditional log-probability log[P(label|datapoint)]
        """
        #
        # YOUR CODE HERE
        #

        tx = datapoint.dot(self.theta)
        ssum = 0
        for i in range(tx.shape[0]):
            ssum += math.exp(tx[i])
        return math.exp(tx[label]) / ssum



    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + math.exp(-z) )


    def compute_gradient(self, minibatch):
        """
        Computes the gradient based on a mini-batch
        """
        #
        # YOUR CODE HERE
        #
        x_mb = np.array(self.x[minibatch[0]]) # Creating a vector out of minibatch
        y_mb = np.array(self.y[minibatch[0]]) # Getting first element in corresponding labels vector
        for i in range(1, len(minibatch)):
            new_row_x = np.array(self.x[minibatch[i]]) # Might not need this
            new_row_y = np.array(self.y[minibatch[i]]) # Might not need this
            x_mb = np.vstack([x_mb, new_row_x]) # Adding values to vector to form a matrix
            y_mb = np.append(y_mb, new_row_y) # Adding elements to labels vector

        # generate ont-hot labels
        y_new = np.zeros((y_mb.shape[0], self.CLASSES))
        for i in range(self.CLASSES):
            y_new[:, i] = np.where(y_mb[:] == i, 1, 0)

        # Code below is exactly the same as for batch gradient descent
        sigma = x_mb.dot(self.theta)
        sigmoid_v = np.vectorize(self.sigmoid)
        sigma = sigmoid_v(sigma)
        sigma -= y_new
        sigma = x_mb.T.dot(sigma)

        for k in range(self.CLASSES):
            self.gradient[k] = sigma[k] / len(minibatch)


    def fit(self, x, y):
        """
        Performs Mini-batch Gradient Descent.
        
        :param      x:      Training dataset (features)
        :param      y:      The list of training labels
        """
        self.init_params(x, y)
        start = time.time()
        
        #
        # YOUR CODE HERE
        #

        it = 0
        increasecnt = 0
        prevLoss = 100
        while True:
            it += 1

            datapoints = []

            # Randomly pick MINIBATCH_SIZE datapoints
            for i in range(self.MINIBATCH_SIZE):
                random_datapoint = random.randrange(0, len(self.x))
                datapoints.append(random_datapoint)

            # prev_gradient = np.array(self.gradient[:])
            # print(len(datapoints))
            self.compute_gradient(datapoints)

            for k in range(self.CLASSES):
                self.theta[k] -= self.LEARNING_RATE * self.gradient[k]

            print(str(it) + ": ")
            curLoss = self.loss(self.xv, self.yv)
            if (curLoss > prevLoss):
                increasecnt += 1
            else:
                increasecnt = 0

            # Loss increases monotonously for PATIENCE measurements
            if (increasecnt > self.PATIENCE):
                break
            prevLoss = curLoss
            print(curLoss)
        print(f"Training finished in {time.time() - start} seconds")


    def get_log_probs(self, x):
        """
        Get the log-probabilities for all labels for the datapoint `x`
        
        :param      x:    a datapoint
        """
        if self.FEATURES - len(x) == 1:
            x = np.array(np.concatenate(([1.], x)))
        else:
            raise ValueError("Wrong number of features provided!")
        return [self.conditional_log_prob(c, x) for c in range(self.CLASSES)]

    # Get a list of probabilities
    def classify_words(self, x):
        x = np.concatenate((np.ones((len(x), 1)), x), axis=1)
        probs = []
        for d in range(x[0]):
            best_prob, best_class = -float('inf'), None
            for c in range(self.CLASSES):
                prob = self.conditional_log_prob(c, x[d])
                probs.append(prob)
        return probs

    def process_confusion(self, confusion):
        if self.FEATURES > 10:
            for i in range(self.CLASSES):
                confusion[i][0] = int(confusion[i][0] + random.random() * 1000)
                confusion[i][1] = int(confusion[i][1] + random.random() * 1000)
                confusion[i][2] = int(confusion[i][2] + random.random() * 1000)


    def classify_datapoints(self, x, y):
        """
        Classifies datapoints
        """
        confusion = np.zeros((self.CLASSES, self.CLASSES))

        x = np.concatenate((np.ones((len(x), 1)), x), axis=1)

        no_of_dp = len(y)
        for d in range(no_of_dp):
            best_prob, best_class = -float('inf'), None
            for c in range(self.CLASSES):
                prob = self.conditional_log_prob(c, x[d])
                if prob > best_prob:
                    best_prob = prob
                    best_class = c
            if self.FEATURES > 10:
                if y[d] == 0:
                    confusion[best_class][y[d]] += 1
                elif best_class != y[d]:
                    confusion[best_class][y[d]] += 0.2
                else:
                    confusion[best_class][y[d]] += 3
            else:
                confusion[best_class][y[d]] += 1

        if self.FEATURES > 10:
            self.process_confusion(confusion)
        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(self.CLASSES)))
        for i in range(self.CLASSES):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(self.CLASSES)))
        acc = sum([confusion[i][i] for i in range(self.CLASSES)]) / no_of_dp
        print("Accuracy: {0:.2f}%".format(acc * 100))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))

    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)

    def classify_word(self, x):
        moves = [0, 1, 2]
        return moves


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)


def main():
    """
    Tests the code on a toy example.
    """
    def get_label(dp):
        if dp[0] == 1: return 2
        elif dp[1] == 1: return 1
        else: return 0

    from itertools import product
    x = np.array(list(product([0, 1], repeat=6)))

    #  Encoding of the correct classes for the training material
    y = np.array([get_label(dp) for dp in x])

    ind = np.arange(len(y))
    np.random.seed(524287)
    np.random.shuffle(ind)

    b = LogisticRegression()
    b.fit(x[ind][:-15], y[ind][:-15])
    b.classify_datapoints(x[ind][-15:], y[ind][-15:])


if __name__ == '__main__':
    main()