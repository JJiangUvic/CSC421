The k-means algorithm is to classify data into k clusters. First, randomly pick k points as the center of
each cluster, calculate the distance of all data to the center, and put the data into the closest center’s
cluster. Next, calculate the mean value of each cluster and use it to update the center. Repeat the steps
above, until the cluster keep the same.
In this task, the training dataset will be split into k clusters as the training model. Each cluster has a
center and all other data in that cluster will be the closest to that center compere to other centers.
For the testing set, each data will find the closest center and sign to that cluster, then check the label of
the cluster, the predict label will be the label which has the most appearance in that cluster.
Task Result:
I test the cross validation for different value of k:
k=3,5,7,11 and fold the number is 5, the graph
shows the average accuracy of each k.
k=3 average accuracy = 0.177
k=5 average accuracy = 0.188
k=7 average accuracy = 0.195
k=11 average accuracy = 0.234
Based on the result, the accuracy increases with k. In other words, a higher k has high accuracy. This
the result is reasonable because the dataset has 10
classes, if the is smaller than k, that means this
the classification process is losing information, the
the cluster cannot provide all possible labels to the data,
which will have a negative effect on the predicted result. When the k increase, the clusters can contain more
information, which can increase the accuracy of the prediction. One class could have more than one
cluster.
For the test set: the final accuracy is : accuracy: 0.2375, when k = 11