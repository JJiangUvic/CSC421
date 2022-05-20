SoftMax is used in the multi-classification process. It transforms the outputs of Neural Networks to the
probability of the input for each class. Use the probability to classify. To process the classification,
the model needs a matrix W and bias b. Before the classification, W and b need to be trained using the
training set. Predict the label for the training set and use the real label to calculate the gradient descent (dw)
and bias(db) and use them to update W and b. in the process, the loss will be calculated.