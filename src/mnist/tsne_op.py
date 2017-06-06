#!/usr/bin/env python

""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow
Example Usage: 
	python draw.py --data_dir=/tmp/draw --read_attn=True --write_attn=True
Author: Eric Jang
"""

import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import LSTMCell
import input_data
import numpy as np
import os
import tsne
import numpy as Math
import pylab as Plot
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn",True, "enable attention for writer")
FLAGS = tf.flags.FLAGS

## MODEL PARAMETERS ## 
data_directory = os.path.join(FLAGS.data_dir, "easy")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
train_data = input_data.read_data_sets(data_directory, one_hot=True).train
A,B = 56,56 # image width,height
img_size = B*A # the canvas size
enc_size = 500 # number of hidden units / output size in LSTM
dec_size = 500
read_n = 12 # read glimpse grid width/height
write_n = 12 # write glimpse grid width/height
read_size = 2*read_n*read_n if FLAGS.read_attn else 2*img_size
rs = np.sqrt(read_size/2).astype(int)
write_size = write_n*write_n if FLAGS.write_attn else img_size
z_size=10 # QSampler output size
T=10 # MNIST generation sequence length
batch_size=train_data._num_examples # training minibatch size
train_iters=10000
learning_rate=1e-3 # learning rate for optimizer
eps=1e-8 # epsilon for numerical stability

## BUILD MODEL ## 

DO_SHARE=None # workaround for variable_scope(reuse=True)

x = tf.placeholder(tf.float32,shape=(batch_size,img_size)) # input (batch_size * img_size)
e=tf.random_normal((batch_size,z_size), mean=0, stddev=1) # Qsampler noise
lstm_enc = LSTMCell(enc_size, (rs/4)*(rs/4)*5+dec_size) # encoder Op
lstm_dec = LSTMCell(dec_size, z_size) # decoder Op
phase_train = tf.placeholder(tf.bool, name='phase_train')

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim]) 
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

from tensorflow.python import control_flow_ops

def batch_norm(x, n_out, phase_train, conv=True, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        if conv:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        else:
            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def filterbank(gx, gy, sigma2,delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx,Fy

def attn_window(scope,h_dec,N):
    with tf.variable_scope(scope,reuse=DO_SHARE):
        params=linear(h_dec,5)
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
    gx=(A+1)/2*(gx_+1)
    gy=(B+1)/2*(gy_+1)
    sigma2=tf.exp(log_sigma2)
    delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N
    return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)

## READ ## 
def read_no_attn(x,x_hat,h_dec_prev):
    return tf.concat(1,[x,x_hat])

def read_attn(x,x_hat,h_dec_prev):
    Fx,Fy,gamma=attn_window("read",h_dec_prev,read_n)
    def filter_img(img,Fx,Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1,B,A])
        glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse*tf.reshape(gamma,[-1,1])
    x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
    x_hat=filter_img(x_hat,Fx,Fy,gamma,read_n)
    return tf.concat(1,[x,x_hat]) # concat along feature axis

read = read_attn if FLAGS.read_attn else read_no_attn

## ENCODE ## 
def encode(state,r,h_dec):
    """
    run LSTM
    state = previous encoder state
    input = cat(read,h_dec_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("encoder",reuse=DO_SHARE):
	"""	
	def lstm_cell(i, o, state):	
	h = tf.get_variable("h", [batch_size, enc_size])
	input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    	forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
	tf.get_variable("w", [x.get_shape()[1], output_dim])
    	update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
    	state = forget_gate * state + input_gate * tf.tanh(update)
    	output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    	return output_gate * tf.tanh(state), state
	"""
        W1=tf.get_variable("W1", [3, 3, 2, 32*2])
        W2=tf.get_variable("W2", [3, 3, 32*2, 64*2]) 
        W3=tf.get_variable("W3", [3, 3, 64*2, 128*2]) 
        W4=tf.get_variable("W4", [1, 1, 128*2, 5]) 


        
        x=tf.reshape(r, [-1,rs,rs,2])
        x=tf.nn.conv2d(x, W1, strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.relu(batch_norm(x, 32*2, phase_train))
        x=tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(batch_norm(x, 64*2, phase_train))
        x=tf.nn.conv2d(x, W3, strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.relu(batch_norm(x, 128*2, phase_train))
        x=tf.nn.conv2d(x, W4, strides=[1, 1, 1, 1], padding='VALID')
        x = tf.nn.relu(batch_norm(x, 5, phase_train))
        input=tf.reshape(x, [-1, (rs/4)*(rs/4)*5])         
	    
        return lstm_enc(tf.concat(1,[input,h_dec]),state)

## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##

def sampleQ(h_enc):
    """
    Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
    mu is (batch,z_size)
    """
    with tf.variable_scope("mu",reuse=DO_SHARE):
        mu=linear(h_enc,z_size)
    with tf.variable_scope("sigma",reuse=DO_SHARE):
        logsigma=linear(h_enc,z_size)
        sigma=tf.exp(logsigma)
    return (mu + sigma*e, mu, logsigma, sigma)

## DECODER ## 
def decode(state,input):
    with tf.variable_scope("decoder",reuse=DO_SHARE):
        
        h_dec, state = lstm_dec(input, state)
        W1=tf.get_variable("W1", [3, 3, 128*2, dec_size])
        W2=tf.get_variable("W2", [3, 3, 64*2, 128*2]) 
        W3=tf.get_variable("W3", [5, 5, 64*2, 64*2]) 
        W4=tf.get_variable("W4", [5, 5, 32*2, 64*2])
        W5=tf.get_variable("W5", [5, 5, 32*2, 32*2])
        W6=tf.get_variable("W6", [5, 5, 1, 32*2]) 

        x=tf.reshape(h_dec, [-1,1,1,dec_size])
        x=tf.nn.conv2d_transpose(x, W1, [batch_size, 3, 3, 128*2], strides=[1, 1, 1, 1], padding='VALID')
        x = tf.nn.relu(batch_norm(x, 128*2, phase_train))
        x=tf.nn.conv2d_transpose(x, W2, [batch_size, 6, 6, 64*2], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.relu(batch_norm(x, 64*2, phase_train))
        x=tf.nn.conv2d_transpose(x, W3, [batch_size, 12, 12, 64*2], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.relu(batch_norm(x, 64*2, phase_train))
        x=tf.nn.conv2d_transpose(x, W4, [batch_size, 12, 12, 32*2], strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(batch_norm(x, 32*2, phase_train))
        x=tf.nn.conv2d_transpose(x, W5, [batch_size, 12, 12, 32*2], strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(batch_norm(x, 32*2, phase_train))
        x=tf.nn.conv2d_transpose(x, W6, [batch_size, 12, 12, 1], strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(batch_norm(x, 1, phase_train))
        return tf.reshape(x, [-1, 12*12]), h_dec, state

## WRITER ## 
def write_no_attn(h_dec):
    with tf.variable_scope("write",reuse=DO_SHARE):
        return linear(h_dec,img_size)

def write_attn(h_dec):
    with tf.variable_scope("writeW",reuse=DO_SHARE):
        w=linear(h_dec,write_size) # batch x (write_n*write_n)
    N=write_n
    w=tf.reshape(w,[batch_size,N,N])
    Fx,Fy,gamma=attn_window("write",h_dec,write_n)
    Fyt=tf.transpose(Fy,perm=[0,2,1])
    wr=tf.batch_matmul(Fyt,tf.batch_matmul(w,Fx))
    wr=tf.reshape(wr,[batch_size,B*A])
    #gamma=tf.tile(gamma,[1,B*A])
    return wr*tf.reshape(1.0/gamma,[-1,1])

write=write_attn if FLAGS.write_attn else write_no_attn

## STATE VARIABLES ## 

cs=[0]*T # sequence of canvases
z=[0]*T
mus,logsigmas,sigmas=[0]*T,[0]*T,[0]*T # gaussian params generated by SampleQ. We will need these for computing loss.
# initial states
h_dec_prev=tf.zeros((batch_size,dec_size))
enc_state=lstm_enc.zero_state(batch_size, tf.float32)
dec_state=lstm_dec.zero_state(batch_size, tf.float32)

## DRAW MODEL ## 

# construct the unrolled computational graph
for t in range(T):
    c_prev = tf.zeros((batch_size,img_size)) if t==0 else cs[t-1]
    x_hat=x-tf.minimum(tf.nn.relu(c_prev),1) # error image
    r=read(x,x_hat,h_dec_prev)
    h_enc, enc_state=encode(enc_state,r,h_dec_prev)
    #h_en = batch_norm(h_enc, 500, phase_train, conv=False)
    z[t],mus[t],logsigmas[t],sigmas[t]=sampleQ(h_enc)
    out,h_dec,dec_state=decode(dec_state,z[t])
    cs[t]=c_prev+write(out) # store results
    h_dec_prev=h_dec
    DO_SHARE=True # from now on, share variables

## LOSS FUNCTION ## 
"""
def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

# reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
x_recons=tf.minimum(tf.nn.relu(cs[-1]),1)

# after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
Lx=tf.reduce_sum(binary_crossentropy(x,x_recons),1) # reconstruction term
Lx=tf.reduce_mean(Lx)

kl_terms=[0]*T
for t in range(T):
    mu2=tf.square(mus[t])
    sigma2=tf.square(sigmas[t])
    logsigma=logsigmas[t]
    kl_terms[t]=0.5*tf.reduce_sum(mu2+sigma2-2*logsigma,1)-T*.5 # each kl term is (1xminibatch)
KL=tf.add_n(kl_terms) # this is 1xminibatch, corresponding to summing kl_terms from 1:T
Lz=tf.reduce_mean(KL) # average over minibatches

cost=Lx+Lz

## OPTIMIZER ## 

optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads=optimizer.compute_gradients(cost)
for i,(g,v) in enumerate(grads):
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
train_op=optimizer.apply_gradients(grads)
"""
## RUN TRAINING ## 

 # binarized (0-1) mnist data

#fetches=[]
#fetches.extend([Lx,Lz,train_op])
Lxs=[0]*train_iters
Lzs=[0]*train_iters


#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

saver = tf.train.Saver() # saves variables learned during training
#tf.initialize_all_variables().run()
#saver.restore(sess, "drawmodel.ckpt") # to restore from model, uncomment this line

saver.restore(sess, "drawmodel.ckpt") # to restore from model, uncomment this line
xtrain=train_data._images # xtrain is (batch_size x img_size)
feed_dict={x:xtrain, phase_train.name: False}
canvases=sess.run(cs,feed_dict)

latent=sess.run(z,feed_dict) # generate some examples
latent=np.array(latent) # T x batch x img_size

latent=latent[1]
print(np.max(latent))
#latent=np.reshape(latent,(T*batch_size,z_size))
Y = tsne.tsne(latent, 2, 50, 5.0);

fig, ax = Plot.subplots()
artists = []
print(Y.shape[0])
print(xtrain.shape[0])
for i, (x0, y0) in enumerate(zip(Y[:,0], Y[:,1])):
    image = xtrain[i%xtrain.shape[0]]
    image = image.reshape(56,56)
    im = OffsetImage(image, zoom=1.0)
    ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
    artists.append(ax.add_artist(ab))
ax.update_datalim(np.column_stack([Y[:,0], Y[:,1]]))
ax.autoscale()


#Plot.scatter(Y[:,0], Y[:,1], 20);
Plot.show();
sess.close()

print('Done drawing! Have a nice day! :)')
