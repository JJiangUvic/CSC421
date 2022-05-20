
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm


def VisualizeLoss(acc, ks):
    '''
    Transfrom a list into a graph
    output: graph
    '''
    plt.plot(ks, acc, marker="o")
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()


class K_Means(object):
    def __init__(self, k=10, tolerance=0.0001, max_iter=300):
        ''' setup
        input:
        k: class number
        tolerance: the minimum difference allowed between previous center and current center
        max_iter: the maximum number of iterations when looking for the center
        '''
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def train(self, dataset, datalabel):
        ''' split the dataset into kth class, each class has a center, find out the most appearance
        label as the predict label
        input: dataset, setlabels 
        '''
        self.centers_ = {}
        self.centers_labels = {}

        # setup the centers and center's labels
        # because the training set is in randomly order
        # therefore pick the first kth image as center is same as using random index
        for i in range(self.k_):
            self.centers_[i] = dataset[i]
            self.centers_labels[i] = None

        # start classifing the dataset set, set a max number of iteration to prevent infinte loop (max_iter_)
        for i in tqdm(range(self.max_iter_)):

            # store the current image to all center's distance and different labels in that class
            self.clf_ = {}
            self.lab_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
                self.lab_[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            # loop all image in the datasetset
            for f in range(dataset.shape[0]):
                current_image = dataset[f]
                current_image_lab = datalabel[f]
                distances = []

                # find the Euclidean Distance to different center
                for center in self.centers_:
                    # Euclidean Distance
                    distances.append(np.linalg.norm(
                        current_image - self.centers_[center]))

                # find the closest center
                classification = distances.index(min(distances))

                # put the this image into the class
                self.clf_[classification].append(current_image)
                # update the different number of labels in that class
                self.lab_[classification][current_image_lab] += 1

            # update the center
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                # calcule the class average
                self.centers_[c] = np.average(self.clf_[c], axis=0)
                # update the class label with the most appearance in that class
                self.centers_labels[c] = self.lab_[
                    c].index(max(self.lab_[c]))

            # ckeck the center is the real center
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                # check the difference between the pre center and current, how differenct
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:  # the different is small enough, break the process
                break

    def predict(self, test_data):
        '''
        find the test_date belongs to which class, and get the class
        label
        input 1 test data
        output the class it belongs to and predict label
        '''
        # calculate the distance to different class's center
        distances = [np.linalg.norm(test_data - self.centers_[center])
                     for center in self.centers_]
        # find the closest center
        index = distances.index(min(distances))
        # return the class and class albel
        return index, self.centers_labels[index]


class kmeans_cifar10(object):
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

    def Cross_validation(self, num_folds=5, k_choices=[3, 5, 7, 11, 20]):
        ''' The cross_validation will be split into num_folds sets and run
        kmeans with different k in k_choices, each k will run num_folds times 
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

                classify = K_Means(k, 0.0001, 300)
                classify.train(X_tr, y_tr)

                curr_correct = 0
                total = 0
                for i in range(X_te.shape[0]):
                    class_, plabel = classify.predict(X_te[i])
                    correct = y_te[i]
                    if correct == plabel:
                        curr_correct += 1
                    total += 1
                k_accuracy[k].append(curr_correct/total)

        acc_list = []
        for k, accuracylist in k_accuracy.items():  # print the accuracy in different k
            print('k value = %d, maximum accuracy = %.3f, minimum accuracy =%.3f, average accuracy = %.3f' % (
                k, max(accuracylist), min(accuracylist), sum(accuracylist)/len(accuracylist)))
            acc_list.append(sum(accuracylist)/len(accuracylist))
            sys.stdout.flush()
        # print(k_accuracy)

        VisualizeLoss(acc_list, k_choices)

    def test_cifar10_testset(self):
        ''' use cifar10 test cases
        '''
        print('train model with test case...')
        sys.stdout.flush()

        k = 11  # best value
        classify = K_Means(k, 0.0001, 300)
        classify.train(self.X_tr, self.y_tr)

        curr_correct = 0
        total = 0
        for i in range(self.X_te.shape[0]):
            class_, plabel = classify.predict(self.X_te[i])
            correct = self.y_te[i]
            if correct == plabel:
                curr_correct += 1
            total += 1
        accuracy = curr_correct/total
        print(f'accuracy: {accuracy},k:{k}')


def main():
    ''' the process will finished in approximate 50 min, in which 40 min for cross 
    validiation and 10 min for testing. (PC performance: cpu 5.0GHz, 32G Ram)

    note: K=11 can maximize the accuracy
    '''
    # create the kmeans object
    km = kmeans_cifar10()

    # cross validiation
    num_folds = 5  # fold numbers
    k_choices = [3, 5, 7, 11, ]  # k values
    km.Cross_validation(num_folds, k_choices)

    # train with CIFAR10 testset
    km.test_cifar10_testset()


if __name__ == '__main__':
    main()
