from .Stats import *
import csv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class LogReg(object):
    # X -> data as [[x1,y1], [x2,y2], ...]
    # Y -> result to predict [0,1,...]
    def __init__(self, X, Y=None, iterations=100, learning_rate=0.01):
        self.n = mylen(X)
        self.size = mylen(X[0])
        self.iterations = 100 if iterations < 100 else iterations
        self.learning_rate = learning_rate
        self.bias = 0
        self.learning_curve = {'accuracy': [], 'loss':[]}
        self.weights = np.random.normal(size=self.n)
        self.x = X
        self.Y = Y
        self.X = [[] for i in range(0, self.size)]
        for i in range(0, self.n):
            # nettoi les données et rescale les valeurs numeriques entre 0 et 1
            X[i] = self.minmaxScaler(X[i])
            # transform les list de la forme[[a1,a1,a3,...], [b1,b2,b3, ...], ...] -> [[a1, b1, ...], [a2, b2, ...], ...]
            for j in range(0, self.size):
                self.X[j].append(X[i][j])

    ##
    ## FEATURES TRANSFORM
    ##

    #	met les valeur de data entre 0 et 1 en gardant la même echelle
    def minmaxScaler(self, data):
        dmin = mymin(data)
        dmax = mymax(data)
        return [((x - dmin) / (dmax - dmin)) for x in data]


    ##
    ## MODEL
    ##

    # retourne une matrice contenant les probabilités de chaque ligne d'appartenir à la classe recherché
    def model(self, X=None):
        X = self.X if X is None else X
        return sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X=None):
        X = self.X if X is None else X
        return np.round(self.model(X))

    # La moyenne des carrés des differances entre la prediction et le dataset d'entrainement
    def loss(self, X=None):
        if self.Y is None:
            return
        X = self.X if X is None else X
        return np.mean((self.predict(X) - self.Y) ** 2)

    def accuracy(self):
        if self.Y is None:
            return
        err = 0
        predictions = self.predict()
        for i in range(0, self.size):
            if predictions[i] != self.Y[i]:
                err += 1
        return 100 - (err / self.size * 100)

    ##
    ## TRAIN
    ##


    def gradient_descent(self):
        if self.Y is None:
            return
        weights = np.zeros(self.weights.shape)
        bias = 0
        for i in range(0, self.size):
            z = self.model(self.X[i])
            y = np.round(z)
            weights += (y - self.Y[i]) * sigmoid_deriv(z) * np.array(self.X[i])
            bias += (y - self.Y[i]) * sigmoid_deriv(z)
        self.weights -= self.learning_rate * weights
        self.bias -= self.learning_rate * bias



    def train(self):
        if self.Y is None:
            return
        fig, (axs) = plt.subplots(3)
        for epoch in tqdm(range(self.iterations)):
            self.gradient_descent()
            if epoch % 2 == 0:
                self.training_plot_animation(axs)
        plt.close()
        plt.show()


    ##
    ## PLOT ANIMATION
    ##

    def training_plot_animation(self, axs):
        self.learning_curve['loss'].append(self.loss())
        self.learning_curve['accuracy'].append(self.accuracy())

        predictions = self.predict()
        x1 = np.linspace(0, 1, 100)
        x2 = (-self.weights[0] * x1 - self.bias) / self.weights[1]
        axs[0].clear()
        axs[0].plot(x1, x2, c='orange', lw=3)
        axs[0].scatter(self.x[0], self.x[1], c=['green' if self.Y[i] == predictions[i] else 'red' for i in range(self.size)])
        axs[0].set_title('Frontiére de désision')


        axs[1].plot(self.learning_curve['loss'])
        axs[1].set_title('Loss')
        axs[2].plot(self.learning_curve['accuracy'])
        axs[2].set_title('Accuracy')
        plt.pause(0.001)
    ##
    ## CSV
    ##

    # save weights and bias in csv file
    def save(self, filepath):
        with open(filepath, 'w+', newline='') as f:
            writer = csv.writer(f)
            row = self.weights.tolist()
            row.append(self.bias)
            writer.writerow(row)

    # load weights and bias from csv file
    def load(self, filepath):
        with open(filepath) as f:
            reader = csv.reader(f)
            for row in reader:
                if mylen(row) > 0:
                    row = list(map(float, row))
                    self.bias = row[-1:][0]
                    self.weights = np.array(row[:-1])
