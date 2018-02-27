import numpy as np
import pandas as pd
from sklearn.utils import resample
import tensorflow as tf
import time


# Import Data Set Using Pandas
data = pd.read_csv('/Users/rodrigotejeida/Desktop/Big Data Project/YearPredictionMSD.txt', header = None)

# Make train/test split
train = data[0:463715]
test  = data[463715:515345]
len(data)==len(train)+len(test)

# Define y and x
xx = data[data.columns[1:92]]
yy = data[data.columns[0:1]]-1900
yyt = np.round(yy/10)
yy = pd.get_dummies(yyt.squeeze()).values
x  = xx[0:463715]
y  = yy[0:463715]
xt = xx[463715:515345]
yt = yy[463715:515345]
x  = x.values
xt = xt.values

# Balance Classes
ys = np.round((train[train.columns[0:1]]-1900)/10)
ys = pd.DataFrame(ys)
df = train[train.columns[1:92]]
df = np.concatenate((ys,df),axis=1)
df = pd.DataFrame(df)

c2  = df[df[0]==2]
c3  = df[df[0]==3]
c4  = df[df[0]==4]
c5  = df[df[0]==5]
c6  = df[df[0]==6]
c7  = df[df[0]==7]
c8  = df[df[0]==8]
c9  = df[df[0]==9]
c10 = df[df[0]==10]
c11 = df[df[0]==11]

c2.size+c3.size+c4.size+c5.size+c6.size+c7.size+c8.size+c9.size+c10.size+c11.size==df.size
max = c10.shape[0]
c2  = resample(c2 ,replace=True,n_samples=max,random_state=735)
c3  = resample(c3 ,replace=True,n_samples=max,random_state=735)
c4  = resample(c4 ,replace=True,n_samples=max,random_state=735)
c5  = resample(c5 ,replace=True,n_samples=max,random_state=735)
c6  = resample(c6 ,replace=True,n_samples=max,random_state=735)
c7  = resample(c7 ,replace=True,n_samples=max,random_state=735)
c8  = resample(c8 ,replace=True,n_samples=max,random_state=735)
c9  = resample(c9 ,replace=True,n_samples=max,random_state=735)
c11 = resample(c11,replace=True,n_samples=max,random_state=735)

df_final = pd.concat((c2,c3,c4,c5,c6,c7,c8,c9,c10,c11),axis=0)

x = df_final[df_final.columns[1:92]]
y = df_final[df_final.columns[0:1]]
y = pd.get_dummies(y.squeeze()).values
x = x.values

# Parameters
# learning_rate = 0.001
learning_rate = 0.001
# training_epochs = 4
training_epochs = 1
# batch_size = 200
batch_size = 200
display_step = 1

# Network Parameters
n_hidden_1 = 90 # 1st layer number of neurons
n_hidden_2 = 90 # 2nd layer number of neurons
n_input    = 90 # MNIST data input (img shape: 28*28)
n_classes  = 10 # MNIST total classes (0-9 digits)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    # 'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    # 'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
def multilayer_perceptron(l):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(l, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Hidden fully connected layer with 256 neurons
    # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Get Batches Method
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

logits = multilayer_perceptron(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
# loss_op = tf.reduce_mean(tf.estimator.LinearRegressor(
#     logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    t0 = time.time()
    # Training cycle
    for epoch in range(training_epochs):
        t00 = time.time()
        avg_cost = 0.
        total_batch = int(y.size/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            if(i%1000==0):
                print('iteration #',i)
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x, batch_y = next_batch(batch_size,x,y)
            # batch_x = batch_x[:,0,:]
            # batch_x = tf.cast(batch_x, tf.float32)
            # batch_y = tf.cast(batch_y, tf.float32)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        t11 = time.time()
        tt  = t11-t00
        print('epoch time',tt)
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

# Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    mse = tf.metrics.mean_absolute_error(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: xt, Y: yt}))
    # MAE = tf.cast(mse, "float")
    # print("MAE:", MAE.eval({X: xt, Y: yt}))
    t1 = time.time()
    total = t1-t0
    print('Total time',total)













































































