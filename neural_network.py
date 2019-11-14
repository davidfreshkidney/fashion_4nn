import numpy as np
import math

# Comment me out!!!
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

"""
    Minibatch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n, d) numpy array where d is the number of features
        y_train (np array) - (n, ) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for 
        testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within 
        limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):
    batchSize = 200         # Default batch size
    n = x_train.shape[0]    # Number of examples

    idxArray = np.arange(n)
    losses = []

    print(">>> Number of examples / batch size: {}/{}".format(n, batchSize))
    print(">>> Running with epoch =", epoch)
    
    for currIter in range(epoch):
        print(">>> Currently at epoch #{}".format(currIter+1))
        if shuffle:
            np.random.shuffle(idxArray)

        totalLoss = 0
        for batchNum in range(int(n / batchSize)):
            currBatchStartIdx = batchNum * batchSize
            X = x_train[idxArray[currBatchStartIdx:currBatchStartIdx+batchSize]]
            y = y_train[idxArray[currBatchStartIdx:currBatchStartIdx+batchSize]]
            totalLoss += four_nn(X, w1, w2, w3, w4, b1, b2, b3, b4, y, False)

        losses += [totalLoss]

    epoch_ax = np.arange(epoch)
    plt.plot(epoch_ax, losses)
    plt.title("epoch vs losses")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    return w1, w2, w3, w4, b1, b2, b3, b4, losses


"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):

    n = x_test.shape[0]

    classifications = four_nn(x_test, w1, w2, w3, w4, b1, b2, b3, b4, y_test, True)

    totalCorrect = 0        # Total correct counter

    # confusionMat = np.zeros((num_classes, num_classes))

    for i in range(len(classifications)):
        if classifications[i] == y_test[i]:
            totalCorrect += 1


    # Plot here

    title = 'Normalized confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_test, classifications)

    classes = np.array(["T-shirt/top","Trouser","Pullover","Dress", "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_test, classifications)]

    # Normalize confusino matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    

    avg_class_rate = totalCorrect / n
    class_rate_per_class = [0.0] * num_classes
    for i in range(num_classes):
        class_rate_per_class[i] = cm[i, i]
    print(">>> avg_class_rate =", avg_class_rate) 
    print(">>> class_rate_per_class =", class_rate_per_class)
    plt.show()
    return avg_class_rate, class_rate_per_class


"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below

"""
def four_nn(X, W1, W2, W3, W4, b1, b2, b3, b4, y, isTest):
    """
        Input:
            X: raw input of shape (n, # of pixels per image)
            W1 - W4: weights for four hidden layers of the NN of shape
            b1 - b4: biases for four hidden layers of the NN
            y: true labels of shape (n, )
            isTest: bool indicating if this is for testing

        Output:
            if this is not a test:
                loss: the loss calculated by cross_entropy
            else:
                classifications: indices of classes of the highest posibility
                 for each token

        Notes:
            The Neural Network must have 4 layers, with (256, 256, 256, and 
             num_classes) nodes per layers.
            You should use a learning rate (lr) of 0.1.

    """

    lr = 0.1     # Learning rate
    loss = 0

    n = y.shape[0]


    Z1, acache1 = affine_forward(X, W1, b1)     # affine_cache: (A, W, b)
    A1, rcache1 = relu_forward(Z1)              # relu_cache: Z
    
    Z2, acache2 = affine_forward(A1, W2, b2)
    A2, rcache2 = relu_forward(Z2)

    Z3, acache3 = affine_forward(A2, W3, b3)
    A3, rcache3 = relu_forward(Z3)

    F, acache4  = affine_forward(A3, W4, b4)

    if isTest:
        classifications = np.argmax(F, axis=1)
        return classifications

    loss, dF = cross_entropy(F, y)

    dA3, dW4, db4 = affine_backward(dF, acache4)
    dZ3 = relu_backward(dA3, rcache3)

    dA2, dW3, db3 = affine_backward(dZ3, acache3)
    dZ2 = relu_backward(dA2, rcache2)

    dA1, dW2, db2 = affine_backward(dZ2, acache2)
    dZ1 = relu_backward(dA1, rcache1)

    dX, dW1, db1  = affine_backward(dZ1, acache1)

    W1 -= lr * dW1
    W2 -= lr * dW2
    W3 -= lr * dW3
    W4 -= lr * dW4

    b1 -= lr * db1
    b2 -= lr * db2
    b3 -= lr * db3
    b4 -= lr * db4

    return loss




"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided 
    as unit_test.py.
    The cache object format is up to you, we will only autograde the computed 
    matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""

def affine_forward(A, W, b):
    """
        Inpiut:
            A: data with shape (n, d)
            W: weights with shape (d, d')
            b: bias with size (d', )

        d is the size of the previous hidden layer, and d' is the size of the 
        current hidden layer
        n is the size of minibatch, aka size of examples
        
        Output:
            Z: affine output of shape (n, d')
            cache: (A, W, b)

    """

    Z = A @ W
    Z += b.T
    cache = (A, W, b)

    return Z, cache


def affine_backward(dZ, cache):
    """
        Input:
            dZ: gradient of Z of shape(n, d')
            cache: (A, W, b)
        Output:
            dA: shape (n, d)
            dW: shape (d, d')
            dB: shape (d', )
    """
    A = cache[0]
    W = cache[1]
    # b = cache[2]

    dA = dZ @ W.T
    dW = A.T @ dZ

    dB = np.sum(dZ, axis=0)

    return dA, dW, dB


def relu_forward(Z):
    """
        Input:
            Z: affine output of shape (n, d')
        Ouput:
            Apply ReLU to Z and output A
            A: Relu Output with shape (n, d')
            cache: Z's copy
    """

    cache = Z
    A = np.maximum(Z, 0)
    # cache = A.copy()

    return A, cache


def relu_backward(dA, cache):
    """
        Input: 
            dA: gradient of A of shape (n, d')
            cache: Z of shape (n, d')

        Output:
            dZ: gradient of Z of shape (n, d')
        
        If Z was zeroed out at a point, then dZ should also be zeroed out,
        otherwise dZ = dA at that point

    """

    Z = cache

    def getFilter(myMat):
        myMat[myMat > 0] = 1
        myMat[myMat <= 0] = 0
        return myMat

    myFilter = getFilter(Z)
    dZ = dA * myFilter

    return dZ


def cross_entropy(F, y):
    """
        Input: 
            F: logits with shape (n, num_classes), n is the num of tokens
            y: actual class label of data with shape (n, )

        Output:
            loss: scalar
            dF: gradient of the logits of shape (n, num_classes)

    """
    n = F.shape[0]
    num_classes = F.shape[1]

    expoF = np.exp(F)

    sumArr = np.zeros((n))

    sumOut = 0
    for i in range(n):
        sumIn = np.sum(expoF[i])
        sumArr[i] = sumIn
        sumOut += F[i, int(y[i])] - math.log(sumIn)

    loss = -sumOut / n

    dF = np.zeros((n, num_classes))

    for i in range(n):
        expSum = sumArr[i]
        for j in range(num_classes):
            oneOrElse = 1 if j == y[i] else 0
            numerator = oneOrElse - expoF[i,j] / expSum
            dF[i,j] = -numerator / n

    return loss, dF
