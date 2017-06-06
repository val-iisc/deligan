
import tensorflow as tf 
import numpy as np
from sklearn import cross_validation
from data_utils import load_CIFAR10
from extract import create_graph, iterate_mini_batches, batch_pool3_features
from datetime import datetime
import matplotlib.pyplot as plt
from tsne import tsne
import os
import sys
import input_data_sketches

#samples = np.load('pure_samples50k.npy').transpose(0,2,3,1)
cifar10_dir ='../../datasets/cifar-10-python/cifar-10-batches-py'   # Change this line to direct to the sketches dataset to test for sketches
def load_pool3_data():
    # Update these file names after you serialize pool_3 values
    X_test_file = 'X_test_1.npy'
    y_test_file = 'y_test_1.npy'
    X_train_file = 'X_train_1.npy'
    y_train_file = 'y_train_1.npy'
    return np.load(X_train_file), np.load(y_train_file), np.load(X_test_file), np.load(y_test_file)

def serialize_cifar_pool3(X,filename):
    print 'About to generate file: %s' % filename
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    X_pool3 = batch_pool3_features(sess,X)
    np.save(filename,X_pool3)

def serialize_data():
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)      # Change this line to take the sketches dataset as input using input_data_sketches.read_data_sets() for testing with sketches
    serialize_cifar_pool3(X_train, 'X_train_1')
    serialize_cifar_pool3(X_test, 'X_test_1')
    np.save('y_train_1',y_train)
    np.save('y_test_1',y_test)

graph=create_graph()    # Comment this line while calculating the inception scores
serialize_data()	# Comment this line while calculating the inception scores
X_sample = np.load('samples50k.npy').transpose(0,2,3,1).astype("float")
X_sample=X_sample*128+127.5
serialize_cifar_pool3(X_sample,'X_sample_1')    	# Comment this line while calculating the inception scores
X_sample_pool3 = np.load('X_sample_1.npy')
print(X_sample_pool3)
#sys.exit()
classes = np.array(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])  # Change this line to test for sketches
X_train_orig, y_train_orig, X_test_orig, y_test_orig = load_CIFAR10(cifar10_dir)    # Change this line to take the sketches dataset as input using input_data_sketches.read_data_sets() for testing with sketches
X_train_pool3, y_train_pool3, X_test_pool3, y_test_pool3 = load_pool3_data()
X_train, X_validation, Y_train, y_validation = cross_validation.train_test_split(X_train_pool3, y_train_pool3, test_size=0.20, random_state=42)


print 'Training data shape: ', X_train_pool3.shape
print 'Training labels shape: ', y_train_pool3.shape
print 'Test data shape: ', X_test_pool3.shape
print 'Test labels shape: ', y_test_pool3.shape
print 'Sample data shape: ', X_sample_pool3.shape
#
# Tensorflow stuff
# #

FLAGS = tf.app.flags.FLAGS
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape'
BOTTLENECK_TENSOR_SIZE = 2048
tf.app.flags.DEFINE_integer('how_many_training_steps', 100,
                            """How many training steps to run before ending.""")
tf.app.flags.DEFINE_float('learning_rate', 0.005,
                          """How large a learning rate to use when training.""")
tf.app.flags.DEFINE_string('final_tensor_name', 'final_result',
                           """The name of the output classification layer in"""
                           """ the retrained graph.""")
tf.app.flags.DEFINE_integer('eval_step_interval', 100,
                            """How often to evaluate the training results.""")



# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def ensure_name_has_port(tensor_name):
    """Makes sure that there's a port number at the end of the tensor name.
    Args:
      tensor_name: A string representing the name of a tensor in a graph.
    Returns:
      The input string with a :0 appended if no port was specified.
    """
    if ':' not in tensor_name:
        name_with_port = tensor_name + ':0'
    else:
        name_with_port = tensor_name
    return name_with_port

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def add_final_training_ops(graph, class_count, final_tensor_name,
                           ground_truth_tensor_name):
    """Adds a new softmax and fully-connected layer for training.
    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.
    The set up for the softmax and fully-connected layers is based on:
    https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
    Args:
      graph: Container for the existing model's Graph.
      class_count: Integer of how many categories of things we're trying to
      recognize.
      final_tensor_name: Name string for the new final node that produces results.
      ground_truth_tensor_name: Name string of the node we feed ground truth data
      into.
    Returns:
      Nothing.
    """
    bottleneck_tensor = graph.get_tensor_by_name(ensure_name_has_port(
        BOTTLENECK_TENSOR_NAME))
    layer_weights = tf.Variable(
        tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001),
        name='final_weights')
    layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
    logits = tf.matmul(bottleneck_tensor, layer_weights,
                       name='final_matmul') + layer_biases
    tf.nn.softmax(logits, name=final_tensor_name)
    ground_truth_placeholder = tf.placeholder(tf.float32,
                                              [None, class_count],
                                              name=ground_truth_tensor_name)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits, ground_truth_placeholder)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy_mean)
    return train_step, cross_entropy_mean

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def add_evaluation_step(graph, final_tensor_name, ground_truth_tensor_name):
    """Inserts the operations we need to evaluate the accuracy of our results.
    Args:
      graph: Container for the existing model's Graph.
      final_tensor_name: Name string for the new final node that produces results.
      ground_truth_tensor_name: Name string for the node we feed ground truth data
      into.
    Returns:
      Nothing.
    """
    result_tensor = graph.get_tensor_by_name(ensure_name_has_port(
        final_tensor_name))
    ground_truth_tensor = graph.get_tensor_by_name(ensure_name_has_port(
        ground_truth_tensor_name))
    correct_prediction = tf.equal(
        tf.argmax(result_tensor, 1), tf.argmax(ground_truth_tensor, 1))
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    return evaluation_step

def encode_one_hot(nclasses,y):
    return np.eye(nclasses)[y]

def do_train(sess,X_input, Y_input, X_validation, Y_validation, X_sample_pool3):
    ground_truth_tensor_name = 'ground_truth'
    mini_batch_size = 1
    n_train = X_input.shape[0]

    graph = create_graph()

    train_step, cross_entropy = add_final_training_ops(
        graph, len(classes), FLAGS.final_tensor_name,
        ground_truth_tensor_name)
    t_vars = tf.trainable_variables()
    final_vars = [var for var in t_vars if 'final_' in var.name]
    saver = tf.train.Saver(final_vars, max_to_keep=10)
    init = tf.initialize_all_variables()
    sess.run(init)

    evaluation_step = add_evaluation_step(graph, FLAGS.final_tensor_name, ground_truth_tensor_name)

    # Get some layers we'll need to access during training.
    bottleneck_tensor = graph.get_tensor_by_name(ensure_name_has_port(BOTTLENECK_TENSOR_NAME))
    ground_truth_tensor = graph.get_tensor_by_name(ensure_name_has_port(ground_truth_tensor_name))
    saver.restore(sess,tf.train.latest_checkpoint(os.getcwd()+"/train/test2/"))
    result_tensor = graph.get_tensor_by_name(ensure_name_has_port(FLAGS.final_tensor_name))
    '''			# Uncomment this line for calculating the inception score
    splits=10
    preds=[]
    print(X_sample_pool3)
    for Xj, Yj in iterate_mini_batches(X_sample_pool3,np.zeros([X_sample_pool3.shape[0],10]),mini_batch_size):
        pred = sess.run(result_tensor,feed_dict={bottleneck_tensor: Xj})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    argmax = preds.argmax(axis=1)
    scores = []
    # Calculating the inception score
    for i in range(splits):
        part = preds[argmax==i]
        logp= np.log(part)
        self = np.sum(part*logp,axis=1)
        cross = np.mean(np.dot(part,np.transpose(logp)),axis=1)
	diff = self - cross
        kl = np.mean(self - cross)
        kl1 = []
        for j in range(splits):
            diffj = diff[(j * diff.shape[0] // splits):((j+ 1) * diff.shape[0] //splits)]
            kl1.append(np.exp(diffj.mean()))
        print("category: %s scores_mean = %.2f, scores_std = %.2f" % (classes[i], np.mean(kl1),np.std(kl1)))
        scores.append(np.exp(kl))
    print("scores_mean = %.2f, scores_std = %.2f" % (np.mean(scores),
                                                     np.std(scores)))
    
    '''			# Uncomment this line for calculating the inception score
    # The block commented out below has to be uncommented for transfer learning and the block above has to be commented
    # '''           # Comment this line when doing transfer learning
    i=0
    epocs = 1
    for epoch in range(epocs):
        shuffledRange = np.random.permutation(n_train)
        y_one_hot_train = encode_one_hot(len(classes), Y_input)
        y_one_hot_validation = encode_one_hot(len(classes), Y_validation)
        shuffledX = X_input[shuffledRange,:]
        shuffledY = y_one_hot_train[shuffledRange]
        for Xi, Yi in iterate_mini_batches(shuffledX, shuffledY, mini_batch_size):
            sess.run(train_step,
                     feed_dict={bottleneck_tensor: Xi,
                                ground_truth_tensor: Yi})
            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                  [evaluation_step, cross_entropy],
                  feed_dict={bottleneck_tensor: Xi,
                             ground_truth_tensor: Yi})
            if (i % 1000)==0:
                saver.save(sess,os.getcwd()+"/train/test2/", global_step=i)
            i+=1
        validation_accuracy=0
        for Xj, Yj in iterate_mini_batches(X_validation, y_one_hot_validation, mini_batch_size):
            validation_accuracy = validation_accuracy + sess.run(evaluation_step,feed_dict={bottleneck_tensor:Xj,ground_truth_tensor:Yj})
        validation_accuracy = validation_accuracy/X_validation.shape[0]
        print('%s: Step %d: Train accuracy = %.1f%%, Cross entropy = %f, Validation accuracy = %.1f%%' %
                    (datetime.now(), i, train_accuracy * 100, cross_entropy_value, validation_accuracy * 100))
    for Xi, Yi in iterate_mini_batches(X_test_pool3,encode_one_hot(len(classes),y_test_pool3), mini_batch_size):
        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_tensor:Xi,ground_truth_tensor:Yi})
    print('Final test accuracy = %.1f%%' % (test_accuracy * 100))
    # '''				# Comment this line when doing transfer learning




gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
do_train(sess,X_train,Y_train,X_validation,y_validation,X_sample_pool3)


