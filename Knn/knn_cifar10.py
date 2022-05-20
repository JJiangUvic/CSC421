import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import sys


def VisualizeLoss(acc, ks):
    '''
    Transfrom a list into a graph
    output: graph
    '''
    plt.plot(ks, acc, marker="o")
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()


class KNearestNeighbor(object):  # KNN algorithm
    def __init__(self):
        '''init None
        '''

        self.X_train = None
        self.y_train = None

    def train(self, X_train, y_train):
        '''KNN has no training prcoess
        train_dataset: pixel data for each image
        train_labels: lables array
        '''

        self.X_train = X_train
        self.y_train = y_train

    def educlidean_distances(self, X_test):
        '''Calculate the educlidean distance convert
        the function (x-y)*2 = X*2 -2xy + y*2
        input test_dataset, numpy.ndarray
        return: distance array, numpy.ndarray
        '''

        # create matrix for storing distance and faster to compute
        dists = np.zeros((X_test.shape[0], self.X_train.shape[0]))
        # calculate -2xy
        value_2xy = np.multiply(X_test.dot(self.X_train.T), -2)
        # calculate x^2
        value_x2 = np.sum(np.square(X_test), axis=1, keepdims=True)
        # calculate y^2
        value_y2 = np.sum(np.square(self.X_train), axis=1)
        # (x-y)*2 = X*2 -2xy + y*2
        dists = value_2xy + value_x2 + value_y2
        return dists

    def predict_label(self, dists, k):
        '''choose the kth nearest distance and use them to predict lable
        input dists: educlidean_distances
        input k: kth nearest
        return: label array
        '''

        y_pred = np.zeros(dists.shape[0])

        for i in range(dists.shape[0]):
            # get the kth closest label
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
            # get the most label appears in the kth closest
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred

    def predict(self, X_test, k):
        ''' get the predict label and compare the orginal label
        '''
        dists = self.educlidean_distances(X_test)
        y_pred = self.predict_label(dists, k)
        return y_pred


class KNNCIAR10(object):

    def __init__(self):
        '''Loading CIFAR10 dataset, if not exist, download from the internet
        '''

        _X = datasets.CIFAR10(
            root='data/', download=True, train=True, transform=ToTensor())
        _Y = datasets.CIFAR10(
            root='data/', download=True, train=False, transform=ToTensor())

        self.X_tr = _X.data.reshape((50000, 32*32*3)).astype("float")
        self.X_te = _Y.data.reshape((10000, 32*32*3)).astype("float")

        self.y_te = np.array(_Y.targets)
        self.y_tr = np.array(_X.targets)

    def Cross_validation(self, num_folds=5, k_choices=[3, 5, 7, 11]):
        ''' The cross_validation will be split into num_folds sets and run
        knn with different k in k_choices, each k will run num_folds times 
        to validate all subset of the training set.
        '''

        print('Start Cross validation:')
        sys.stdout.flush()
        k_accuracy = {}
        # split into num_folds subsets
        X_train_folds = np.array_split(self.X_tr, num_folds)
        y_train_folds = np.array_split(self.y_tr, num_folds)

        for k in k_choices:  # first loop to iterate all ks
            k_accuracy[k] = []
            print(f'current calculating: k={k}')
            sys.stdout.flush()

            # second loop to iterate all subsets as test case
            for index in range(num_folds):
                X_te = X_train_folds[index]
                y_te = y_train_folds[index]

                X_tr = np.reshape(X_train_folds[:index] + X_train_folds[index + 1:],
                                  (self.X_tr.shape[0] * (num_folds - 1) // num_folds, -1))
                y_tr = np.reshape(y_train_folds[:index] + y_train_folds[index + 1:],
                                  (self.X_tr.shape[0] * (num_folds - 1) // num_folds))

                # claculate the knn accuracy
                classify = KNearestNeighbor()
                classify.train(X_tr, y_tr)
                y_te_pred = classify.predict(X_te, k=k)
                accuracy = np.sum(y_te_pred == y_te) / float(X_te.shape[0])
                k_accuracy[k].append(accuracy)

        acc_list = []

        for k, accuracylist in k_accuracy.items():  # print the accuracy in different k
            print('k value = %d, maximum accuracy = %.3f, minimum accuracy =%.3f, average accuracy = %.3f' % (
                k, max(accuracylist), min(accuracylist), sum(accuracylist)/len(accuracylist)))
            acc_list.append(sum(accuracylist)/len(accuracylist))
            sys.stdout.flush()

        VisualizeLoss(acc_list, k_choices)

    def test_cifar10_testset(self):
        ''' use cifar10 test cases
        '''
        print('train model with test case...')
        sys.stdout.flush()

        classify = KNearestNeighbor()
        classify.train(self.X_tr, self.y_tr)
        # base on the result of cross validation, k=7 can maximize the accuracy
        k = 7
        y_te_pred = classify.predict(self.X_te, k)
        accuracy = np.sum(y_te_pred == self.y_te) / float(self.X_te.shape[0])
        print(k, 'accuracy is', accuracy)
        sys.stdout.flush()


def main():
    ''' the process will finished in approximate 13 min, in which 12 min for cross 
    validiation and 1 min for testing. (PC performance: cpu 5.0GHz, 32G Ram)

    note: K=7 can maximize the accuracy
    '''
    # create the knn object
    kf = KNNCIAR10()

    # cross validiation
    num_folds = 5  # fold numbers
    k_choices = [3, 5, 7, 10, 11, 20]  # k values
    kf.Cross_validation(num_folds, k_choices)

    # train with CIFAR10 testset
    kf.test_cifar10_testset()
    print('done')


if __name__ == '__main__':
    main()
