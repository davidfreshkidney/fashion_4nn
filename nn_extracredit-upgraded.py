import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 		#for training on gpu


print(">>> Loading data")

num_classes = 10
labels = np.array(["T-shirt/top","Trouser","Pullover","Dress", "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}

x_train = np.load("data/x_train.npy")
x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
x_train = x_train.reshape(-1,28,28,1)
y_train = np.load("data/y_train.npy")
y_train = tf.compat.v1.Session().run(tf.one_hot(y_train, num_classes))

x_test = np.load("data/x_test.npy")
x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
x_test = x_test.reshape(-1,28,28,1)
y_test_orginal = np.load("data/y_test.npy")
y_test = tf.compat.v1.Session().run(tf.one_hot(y_test_orginal, num_classes))

print(">>> Training data images have dimension {}".format(x_train.shape))
print(">>> Training data labels have dimension {}".format(y_train.shape))
print(">>> Testing data images have dimension {}".format(x_test.shape))
print(">>> Testing data labels have dimension {}".format(y_test.shape))


epoch = 200   	# Training iterations
η = 0.001     	# Learning rate
batchSize = 200

n_input = 28
dimension = 784
n_classes = 10

x = tf.compat.v1.placeholder("float", [None, 28,28,1])
y = tf.compat.v1.placeholder("float", [None, n_classes])


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(input=x, filters=W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool2d(input=x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


weights = {
    'wc1': tf.compat.v1.get_variable('W0', shape=(3,3,1,32), initializer=tf.compat.v1.glorot_uniform_initializer(), use_resource=False), 
    'wc2': tf.compat.v1.get_variable('W1', shape=(3,3,32,64), initializer=tf.compat.v1.glorot_uniform_initializer(), use_resource=False), 
    'wc3': tf.compat.v1.get_variable('W2', shape=(3,3,64,128), initializer=tf.compat.v1.glorot_uniform_initializer(), use_resource=False), 
    'wd1': tf.compat.v1.get_variable('W3', shape=(4*4*128,128), initializer=tf.compat.v1.glorot_uniform_initializer(), use_resource=False), 
    'out': tf.compat.v1.get_variable('W6', shape=(128,n_classes), initializer=tf.compat.v1.glorot_uniform_initializer(), use_resource=False), 
}

biases = {
    'bc1': tf.compat.v1.get_variable('B0', shape=(32), initializer=tf.compat.v1.glorot_uniform_initializer(), use_resource=False),
    'bc2': tf.compat.v1.get_variable('B1', shape=(64), initializer=tf.compat.v1.glorot_uniform_initializer(), use_resource=False),
    'bc3': tf.compat.v1.get_variable('B2', shape=(128), initializer=tf.compat.v1.glorot_uniform_initializer(), use_resource=False),
    'bd1': tf.compat.v1.get_variable('B3', shape=(128), initializer=tf.compat.v1.glorot_uniform_initializer(), use_resource=False),
    'out': tf.compat.v1.get_variable('B4', shape=(10), initializer=tf.compat.v1.glorot_uniform_initializer(), use_resource=False),
}


def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.stop_gradient(y)))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=η).minimize(cost)

# Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
predictions = tf.argmax(input=pred, axis=1)
# print(predictions.shape)
correct_prediction = tf.equal(predictions, tf.argmax(input=y, axis=1))

# calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))


# Initializing the variables
init = tf.compat.v1.global_variables_initializer()


with tf.compat.v1.Session() as sess:
    sess.run(init) 
    train_loss = []
    train_accuracy = []
    summary_writer = tf.compat.v1.summary.FileWriter('./Output', sess.graph)
    for i in range(epoch):
        print(">>> Currently at epoch #{}".format(i+1))
        # print(">>>", len(x_train)/batchSize, int(len(x_train)/batchSize))
        totalLoss = 0
        totalAcc = 0.0
        for batchNum in range(int(x_train.shape[0] / batchSize)):
            currBatchStartIdx = batchNum * batchSize
            batch_x = x_train[currBatchStartIdx:currBatchStartIdx+batchSize]
            batch_y = y_train[currBatchStartIdx:currBatchStartIdx+batchSize]
            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
            totalLoss += loss
            totalAcc += acc
        train_loss.append(totalLoss)
        avgAccuracy = totalAcc / (x_train.shape[0] / batchSize)
        train_accuracy.append(avgAccuracy)
        print(">>> Loss:", totalLoss)
        print(">>> Accuracy:", avgAccuracy)
    print(">>> Training complete")

    # Calculate accuracy for all 10000 mnist test images
    predArr, test_acc,valid_loss = sess.run([predictions, accuracy, cost], feed_dict={x: x_test,y : y_test})
    print(">>> Testing loss:", valid_loss)
    # print(">>> Testing Accuracy:","{:.5f}".format(test_acc))

    plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend()
    plt.show()

    # Plot here

    title = 'Normalized confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_test_orginal, predArr)

    classes = np.array(["T-shirt/top","Trouser","Pullover","Dress", "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_test_orginal, predArr)]

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
    
    class_rate_per_class = [0.0] * num_classes
    for i in range(num_classes):
        class_rate_per_class[i] = cm[i, i]
    print(">>> avg_class_rate =", test_acc) 
    print(">>> class_rate_per_class =", class_rate_per_class)
    plt.show()

    summary_writer.close()

