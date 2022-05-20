import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


def VisualizeLoss(loss_history):
    '''
    Transfrom a list into a graph
    input: loss_history (list)
    output: graph
    '''
    plt.plot(loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()


def VisualizeW(softmax):
    '''
    show the W for different classes
    input: W from the softmax algorithm
    output: image for different classes
    '''
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    # Visualize the learned weights for each class
    w = softmax.W
    w = w.reshape(10, 32, 32, 3)  # reshap the data for display
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        wimg = 255.0 * (w[i, :, :, :].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()


class Softmax(object):

    def __init__(self):
        ''' 
        set the default for W and b, when get the shape of the training set
        then initialize W and b
        '''
        self.W = None
        self.b = None

    def train(self, X, y, reg, learning_rate, batch_num, num_iter, output):
        '''
        train the W and b with the traning set, and record the loss
        '''
        num_train = X.shape[0]
        num_dim = X.shape[1]

        # class start at 0 so the num of class need +1
        num_classes = np.max(y) + 1

        # initialize W and b
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_classes, num_dim)
        if self.b is None:
            self.b = 0.001 * np.random.randn(batch_num, num_classes)

        loss_history = []
        for i in range(num_iter):
            # Mini batch
            sample_index = np.random.choice(
                num_train, batch_num, replace=False)
            X_batch = X[sample_index, :]
            y_batch = y[sample_index]

            # claculate the loss, gradient desecent(dw) and bias(db)
            loss, dw, db = self.cross_entropy_loss(X_batch, y_batch, reg)
            loss_history.append(loss)  # add ing loss to the list

            # update W and bias
            self.W -= learning_rate * dw
            self.b -= learning_rate * db

            # print current iteration and loss
            if output and i % 100 == 0:
                print('Iteration %d / %d: loss %f' % (i, num_iter, loss))

        return loss_history

    def predict(self, X):
        '''
        W*X calculate the prob for each class, and choose the highest one
        '''
        scores = X.dot(self.W.T)
        y_pred = np.zeros(X.shape[0])
        y_pred = np.argmax(scores, axis=1)

        return y_pred

    def cross_entropy_loss(self, X, y, reg):
        '''
        X: N * D, y: N
        claculate the loss, gradient descent(dw) and bias(db)
        output: loss, gradient desecent(dw) and bias(db)
        '''

        num_train = X.shape[0]
        scores = X.dot(self.W.T)+self.b  # W*X+b

        # Max trick for the softmax, preventing infinite values
        scores = (scores - np.matrix(np.max(scores, axis=1)).T).getA()
        exp_scores = np.exp(scores)

        # softmax function
        pro_scores = (
            exp_scores / np.matrix(np.sum(exp_scores, axis=1)).T).getA()

        # calculat the dw and db
        ground_true = np.zeros(scores.shape)
        ground_true[range(num_train), y] = 1
        # loss
        loss = -1 * np.sum(ground_true * np.log(pro_scores)) / \
            num_train + 0.5 * reg * np.sum(self.W * self.W)
        # dw
        dw = -1 * (ground_true - pro_scores).T.dot(X) / \
            num_train + reg * self.W
        # db
        db = (ground_true - pro_scores)/num_train + reg * self.b

        return loss, dw, db


class SOFTMAXCIAR10(object):
    '''
    This class will use CIFAR10 dataset to training a softmax model
    '''

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

    def validateSoftmax(self):
        ''' 
        split the training set, do a validiation rest to test different values 
        of learning rate and regularizationstrengths and find the best
        '''
        print('start validiate learning rates and regularization strengths...')

        # inner class for testing, pass the dataset to the inner class object
        s = self.TestingSoftmax(self)

        num_train = 49000  # no greater than 50000
        num_val = 1000
        learning_rates = [1e-7, 5e-5]  # could add more
        regularization_strengths = [5e4, 1e5]  # could add more

        # spilt the training set into the vlaidiation size
        s.pre_process_data(num_train, num_val)

        # find the best value
        best_learning_rate, best_reg = s.Testing(
            learning_rates, regularization_strengths)

        print(
            f"Done! the best_learning_rate:{best_learning_rate} and the best_reg:{best_reg}")

    def testCIFAR10(self, batch_num=200, num_iter=1500, learning_rate=1e-7, regularization_strengths=5e4):
        '''
        Use the CIFAR10 training set and testing set to training and predict
        '''
        softmax = Softmax()
        loss_history = softmax.train(self.X_tr, self.y_tr, regularization_strengths,
                                     learning_rate, batch_num, num_iter, True)
        y_pred = softmax.predict(self.X_te)
        acc = np.mean(y_pred == self.y_te)

        print(f"The accuracy is:{acc}")

        # print a loss graph
        VisualizeLoss(loss_history)

        # print image for each class
        VisualizeW(softmax)

    class TestingSoftmax(object):
        ''' split the training set smae as the cross validiation,
        and try different learning_rates and regularization_strengths
        to find the best pair
        '''

        def __init__(self, softmax):  # get the sets from outter class
            self.X_tr = softmax.X_tr
            self.X_te = softmax.X_te
            self.y_tr = softmax.y_tr
            self.y_te = softmax.y_te

        def pre_process_data(self, num_train, num_val):  # split the data set
            mask = range(num_train, num_train + num_val)
            self.X_val = self.X_tr[mask]
            self.y_val = self.y_tr[mask]
            self.X_tr = self.X_tr[:num_train]
            self.y_tr = self.y_tr[:num_train]

            self.X_tr = np.reshape(self.X_tr, (self.X_tr.shape[0], -1))
            self.X_te = np.reshape(self.X_te, (self.X_te.shape[0], -1))
            self.X_val = np.reshape(self.X_val, (self.X_val.shape[0], -1))

            mean_image = np.mean(self.X_tr, axis=0)
            self.X_tr -= mean_image
            self.X_val -= mean_image
            self.X_te -= mean_image

        def Testing(self, learning_rates, regularization_strengths):
            best_val = -1
            best_parameters = None

            # try different combination of learning_rates and regularization_strengths
            for i in learning_rates:
                for j in regularization_strengths:
                    softmax = Softmax()
                    softmax.train(self.X_tr, self.y_tr, j, i, 200, 1500, False)
                    y_pred = softmax.predict(self.X_val)
                    acc = np.mean(y_pred == self.y_val)
                    print(
                        f'learning_rates:{i}, regularization_strengths:{j}, acc:{acc}')
                    if acc > best_val:
                        best_val = acc
                        best_parameters = (i, j)

            return best_parameters[0], best_parameters[1]


def main():
    batch_num = 400
    num_iter = 4800
    learning_rate = 1e-7
    regularization_strengths = 5e4

    s = SOFTMAXCIAR10()

    # test different values of learning rate and regularizationstrengths
    # modify in line 169 170
    s.validateSoftmax()

    # test the CIFAR10 training set and testing set
    s.testCIFAR10(batch_num, num_iter, learning_rate, regularization_strengths)

    print('done')


if __name__ == '__main__':
    main()
