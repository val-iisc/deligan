# This is the code for experiments performed on the MNIST dataset for the DeLiGAN model. Minor adjustments in 
# the code as suggested in the comments can be done to test GAN. Corresponding details about these experiments 
# can be found in section 5.3 of the paper and the results showing the outputs can be seen in Fig 4.

import tensorflow as tf
import numpy as np
from ops import *
from utils import *
import os
import time
from random import randint
import cv2
import matplotlib.pylab as Plot
import tsne
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib
import numpy as Math
import sys
from tensorflow.contrib.layers import batch_norm

data_dir='../datasets/mnist/'
results_dir='../results/mnist/'
phase_train = tf.placeholder(tf.bool, name = 'phase_train')
def Minibatch_Discriminator(input, num_kernels=100, dim_per_kernel=5, init=False, name='MD'):
    num_inputs=df_dim*4
    theta = tf.get_variable(name+"/theta",[num_inputs, num_kernels, dim_per_kernel], initializer=tf.random_normal_initializer(stddev=0.05))
    log_weight_scale = tf.get_variable(name+"/lws",[num_kernels, dim_per_kernel], initializer=tf.constant_initializer(0.0))
    W = tf.mul(theta, tf.expand_dims(tf.exp(log_weight_scale)/tf.sqrt(tf.reduce_sum(tf.square(theta),0)),0))
    W = tf.reshape(W,[-1,num_kernels*dim_per_kernel])
    x = input
    x=tf.reshape(x, [batchsize,num_inputs])
    activation = tf.matmul(x, W)
    activation = tf.reshape(activation,[-1,num_kernels,dim_per_kernel])
    abs_dif = tf.multiply(tf.reduce_sum(tf.abs(tf.subtract(tf.expand_dims(activation,3),tf.expand_dims(tf.transpose(activation,[1,2,0]),0))),2), 
                                                1-tf.expand_dims(tf.constant(np.eye(batchsize),dtype=np.float32),1))
    f = tf.reduce_sum(tf.exp(-abs_dif),2)/tf.reduce_sum(tf.exp(-abs_dif))  
    print(f.get_shape())
    print(input.get_shape())
    return tf.concat(1,[x, f])

def linear(x,output_dim, name="linear"):

    w=tf.get_variable(name+"/w", [x.get_shape()[1], output_dim])
    b=tf.get_variable(name+"/b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def fc_batch_norm(x, n_out, phase_train, name='bn'):
        beta = tf.get_variable(name + '/fc_beta', shape=[n_out], initializer=tf.constant_initializer())
        gamma = tf.get_variable(name + '/fc_gamma', shape=[n_out], initializer=tf.random_normal_initializer(1., 0.02))
        batch_mean, batch_var = tf.nn.moments(x, [0], name=name + '/fc_moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
        return normed

def global_batch_norm(x, n_out, phase_train, name='bn'):
    beta = tf.get_variable(name + '/beta', shape=[n_out], initializer=tf.constant_initializer(0.))
    gamma = tf.get_variable(name + '/gamma', shape=[n_out], initializer=tf.random_normal_initializer(1., 0.02))
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name=name + '/moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return normed

def conv(x,  Wx, Wy,inputFeatures, outputFeatures, stridex=1, stridey=1, padding='SAME', transpose=False, name='conv'):
    w = tf.get_variable(name+"/w",[Wx, Wy, inputFeatures, outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable(name+"/b",[outputFeatures], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(x, w, strides=[1,stridex,stridey,1], padding=padding) + b
    return conv

def convt(x, outputShape, Wx=3, Wy=3, stridex=1, stridey=1, padding='SAME', transpose=False, name='convt'):
    w = tf.get_variable(name+"/w",[Wx, Wy, outputShape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable(name+"/b",[outputShape[-1]], initializer=tf.constant_initializer(0.0))
    convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=[1,stridex,stridey,1], padding=padding) +b
    return convt

def discriminator(image, Reuse=False):
    with tf.variable_scope('disc', reuse=Reuse):
        image = tf.reshape(image, [-1, 28, 28, 1])
        h0 = lrelu(conv(image, 5, 5, 1, df_dim, stridex=2, stridey=2, name='d_h0_conv'))
        h1 = lrelu( batch_norm(conv(h0, 5, 5, df_dim,df_dim*2,stridex=2,stridey=2,name='d_h1_conv'), decay=0.9, scale=True, updates_collections=None, is_training=phase_train, reuse=Reuse, scope='d_bn1'))
        h2 = lrelu(batch_norm(conv(h1, 3, 3, df_dim*2, df_dim*4, stridex=2, stridey=2,name='d_h2_conv'), decay=0.9,scale=True, updates_collections=None, is_training=phase_train, reuse=Reuse, scope='d_bn2'))
        h3 = tf.nn.max_pool(h2, ksize=[1,4,4,1], strides=[1,1,1,1],padding='VALID')
        h6 = tf.reshape(h2,[-1, 4*4*df_dim*4])
        h7 = Minibatch_Discriminator(h3, num_kernels=df_dim*4, name = 'd_MD')
        h8 = dense(tf.reshape(h7, [batchsize, -1]), df_dim*4*2, 1, scope='d_h8_lin')
        return tf.nn.sigmoid(h8), h8 

def generator(z):
    with tf.variable_scope('gen'):
        h0 = tf.reshape(tf.nn.relu(fc_batch_norm(linear(z, gf_dim*4*4*4, name='g_h0'), gf_dim*4*4*4, phase_train, 'g_bn0')), [-1, 4, 4, gf_dim*4])
        h1 = tf.nn.relu(global_batch_norm(convt(h0,[batchsize, 7, 7, gf_dim*2],3, 3, 2, 2, name='g_h1'), gf_dim*2, phase_train, 'g_bn1'))
        h3 = tf.nn.relu(global_batch_norm(convt(h1,[batchsize, 14, 14,gf_dim],5, 5, 2, 2, name='g_h3'), gf_dim, phase_train, 'g_bn3'))
        h4 = tf.tanh(convt(h3,[batchsize, 28, 28, 1], 5, 5, 2, 2, name='g_h4'))
        return h4

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    batchsize = 50
    imageshape = [28*28]
    z_dim = 30
    gf_dim = 16
    df_dim = 16
    learningrate = 0.0005
    beta1 = 0.5

    images = tf.placeholder(tf.float32, [batchsize] + imageshape, name="real_images")
    z = tf.placeholder(tf.float32, [None, z_dim], name="z")
    lr1 = tf.placeholder(tf.float32, name="lr")
    # Our Mixture Model modifications
    zin = tf.get_variable("g_z", [batchsize, z_dim],initializer=tf.random_uniform_initializer(-1,1))
    zsig = tf.get_variable("g_sig", [batchsize, z_dim],initializer=tf.constant_initializer(0.2))
    inp = tf.add(zin,tf.multiply(z,zsig))
    # inp = z     				# Uncomment this line when training/testing baseline GAN
    G = generator(inp)
    D_prob, D_logit = discriminator(images)

    D_fake_prob, D_fake_logit = discriminator(G, Reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit, tf.ones_like(D_logit)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logit, tf.zeros_like(D_fake_logit)))

    sigma_loss = tf.reduce_mean(tf.square(zsig-1))/3    # sigma regularizer
    gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logit, tf.ones_like(D_fake_logit)))
    dloss = d_loss_real + d_loss_fake 

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    data = np.load(data_dir + 'mnist.npz')
    trainx = np.concatenate([data['x_train']], axis=0)
    trainy = np.concatenate([data['y_train']], axis=0)
    trainx = trainx*2-1
    print(trainy.shape)
    data = []
    # Uniformly sampling 50 images per category from the dataset
    for i in range(10):
        print(np.sum(trainy==i))
        train = trainx[trainy==i]
        print(train.shape)
        data.append(train[-50:])
    data = np.array(data)
    print(data.shape)
    data = np.reshape(data,[-1,28*28])

    d_optim = tf.train.AdamOptimizer(lr1, beta1=beta1).minimize(dloss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(lr1, beta1=beta1).minimize(gloss + sigma_loss, var_list=g_vars)
    tf.initialize_all_variables().run()

    saver = tf.train.Saver(max_to_keep=10)

    counter = 1
    start_time = time.time()
    data_size = data.shape[0]
    display_z = np.random.uniform(-1.0, 1.0, [batchsize, z_dim]).astype(np.float32)

    seed = 1
    rng = np.random.RandomState(seed)
    train = True
    thres=1.0      # used to balance gan training
    count1=0
    count2=0
    t1=0.70

    if train:
        # saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+"../results/mnist/train/"))
        # training a model
        for epoch in xrange(4000):
            batch_idx = data_size/batchsize
            batch = data[rng.permutation(data_size)]
            lr = learningrate * (np.minimum((4 - epoch/1000.), 3.)/3)
            for idx in xrange(batch_idx):
                batch_images = batch[idx*batchsize:(idx+1)*batchsize]
                batch_z = np.random.uniform(-1.0, 1.0, [batchsize, z_dim]).astype(np.float32)
                if count1>3:
                    thres=min(thres+0.003, 1.0)
                    count1=0
                    print('gen', thres)
                if count2<-1:
                    thres=max(thres-0.003, t1)
                    count2=0
                    print('disc', thres)

                for k in xrange(5):
                    batch_z = np.random.normal(0, 1.0, [batchsize, z_dim]).astype(np.float32)
                    if gloss.eval({z: batch_z, phase_train.name:False})>thres:
                        sess.run([g_optim],feed_dict={z: batch_z, lr1:lr, phase_train.name:True})
                        count1+=1
                        count2=0
                    else:
                        sess.run([d_optim],feed_dict={ images: batch_images, z: batch_z, lr1:lr, phase_train.name:True})
                        count2-=1
                        count1=0
                counter += 1
                if counter % 300 == 0:
                    # Saving 49 randomly generated samples
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, "  % (epoch, idx, batch_idx, time.time() - start_time,))
                    sdata = sess.run(G,feed_dict={ z: batch_z, phase_train.name:False})
                    sdata = sdata.reshape(sdata.shape[0], 28, 28, 1)/2.+0.5
                    sdata = merge(sdata[:49],[7,7])
                    sdata = np.array(sdata*255.,dtype=np.int)
                    cv2.imwrite(results_dir + "/" + str(counter) + ".png", sdata)
                    errD_fake = d_loss_fake.eval({z: display_z, phase_train.name:False})
                    errD_real = d_loss_real.eval({images: batch_images, phase_train.name:False})
                    errG = gloss.eval({z: display_z, phase_train.name:False})
                    sigloss = sigma_loss.eval()
                    print('D_real: ', errD_real)
                    print('D_fake: ', errD_fake)
                    print('G_err: ', errG)
                    print('sigloss: ', sigloss)
                if counter % 2000 == 0:
                    # Calculating the Nearest Neighbours corresponding to the generated samples
                    sdata = sess.run(G,feed_dict={ z: display_z, phase_train.name:False})
                    sdata = sdata.reshape(sdata.shape[0], 28*28)
                    NNdiff = np.sum(np.square(np.expand_dims(sdata,axis=1) - np.expand_dims(data,axis=0)),axis=2)
                    NN = data[np.argmin(NNdiff,axis=1)]
                    sdata = sdata.reshape(sdata.shape[0], 28, 28, 1)/2.+0.5
                    NN = np.reshape(NN, [batchsize, 28, 28, 1])/2.+0.5
                    sdata = merge(sdata[:49],[7,7])
                    NN = merge(NN[:49],[7,7])
                    sdata = np.concatenate([sdata, NN], axis=1)
                    sdata = np.array(sdata*255.,dtype=np.int)
                    cv2.imwrite(results_dir + "/NN" + str(counter) + ".png", sdata)#gan_1nin_8gfdim_floss_alpha1_z15

                    # Plotting the latent space using tsne
                    z_Mog = zin.eval()#display_z
                    gen = G.eval({z:display_z,  phase_train.name:False})
                    Y = tsne.tsne(z_Mog, 2, z_dim, 10.0);
                    Plot.scatter(Y[:,0], Y[:,1])
                    xtrain = gen.copy()
                    fig, ax = Plot.subplots()
                    artists = []
                    for i, (x0, y0) in enumerate(zip(Y[:,0], Y[:,1])):
                        image = xtrain[i%xtrain.shape[0]]
                        image = image.reshape(28,28)
                        im = OffsetImage(image, zoom=1.0)
                        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
                        artists.append(ax.add_artist(ab))
                    ax.update_datalim(np.column_stack([Y[:,0], Y[:,1]]))
                    ax.autoscale()
                    Plot.scatter(Y[:,0], Y[:,1], 20);
                    fig.savefig(results_dir + "/plot" + str(counter) + ".png")
                    saver.save(sess,os.getcwd()+"../results/mnist/train/", global_step=counter)
    else:
        #Generating samples from a saved model
        saver.restore(sess,tf.train.latest_checkpoint(os.getcwd()+"../results/mnist/train/"))
        samples=[]
        for i in range(100):
            batch_z = np.random.uniform(-1, 1, [batchsize, z_dim]).astype(np.float32)
            sdata = sess.run(G,feed_dict={z: batch_z, phase_train.name:False})
            sdata = sdata.reshape(sdata.shape[0], 28, 28, 1)/2.+0.5
            sdata = sdata*255.
            samples.append(sdata)
        samples1 = np.concatenate(samples,0)
        np.save(results_dir + '/MNIST_samples5k.npy',samples1)
        print("samples saved")
        sys.exit()

