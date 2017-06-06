# This is the code for experiments performed on the Toy dataset for DeLiGAN model. Minor adjustments 
# in the code as suggested in the comments can be done to test GAN and other baseline models such as 
# GAN++, Nx-GAN, MoE-GAN, Ensemble-GAN. Corresponding details about these experiments can be found 
# in section 5.2 of the paper and the results showing the outputs can be seen in Fig 3.

import tensorflow as tf
import numpy as np
import os
import time
from random import randint
import cv2
import matplotlib.pylab as Plot
batchsize=50
results_dir='../results/toy'
def linear(x,output_dim, name="linear"):
    """ Linear Layer for 2d input x """
    w=tf.get_variable(name+"/w", [x.get_shape()[1], output_dim])
    b=tf.get_variable(name+"/b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def discriminator(image, reuse=False):
    """ Discriminator function description """
    with tf.variable_scope('disc', reuse=reuse):
        h0 = tf.tanh(linear(image,df_dim,'d_l1'))
        h1 = linear(h0, 1, 'd_l2')
        return tf.nn.sigmoid(h1), h1

def generator(z, n):
    """ Generator function description """
    with tf.variable_scope('gen'+str(n)):
        #z = tf.tanh(linear(z, batchsize,'g_l0')) 		# Uncomment for testing GAN++ model
	h1 = tf.tanh(linear(z, gf_dim,'g_l1'))	
        h2 = linear(h1, 2, 'g_l2')
        return tf.nn.tanh(h2)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    imageshape = [2]
    z_dim = 2
    gf_dim = 32
    #gf_dim = 32*50 						# Uncomment this line for testing Nx-GAN
    df_dim = 32
    learningrate = 0.0001
    beta1 = 0.5
	
    # Taking Inputs for the graph as placeholders
    images = tf.placeholder(tf.float32, [batchsize] + imageshape, name="real_images")
    z = tf.placeholder(tf.float32, [None, z_dim], name="z")
    lr1 = tf.placeholder(tf.float32, name="lr")

    
    zin = tf.get_variable("g_z", [batchsize, z_dim],initializer=tf.random_uniform_initializer(-1,1))
    zsig = tf.get_variable("g_sig", [batchsize, z_dim],initializer=tf.constant_initializer(0.02))
    inp = tf.add(zin,tf.mul(z,zsig))				#Uncomment this line for testing the DeliGAN
    #moe = tf.eye(batchsize)					#Uncomment this line for testing the MoE-GAN
    #inp = tf.concat_v2([moe, z],1) 				#Uncomment this line for testing the MoE-GAN

    # Calling the generator Function for different Models
    #G = generator(z[:1],0)					#Uncomment this line when testing Ensemble-GAN
    G = generator(inp,0) 					#Uncomment this line for testing DeliGAN, MoE-GAN
    #G = generator(z,0) 					#Uncomment this line for testing GAN and Nx-GAN
    #for n in range(batchsize-1):				#Uncomment this line when testing Ensemble-GAN
    #	g = generator(z[n+1:n+2],n+1)				#Uncomment this line when testing Ensemble-GAN
    #G = tf.concat_v2([g,G],0)					#Uncomment this line when testing Ensemble-GAN
    
    lab = tf.where(G[:,0]<0)
    D_prob, D_logit = discriminator(images)
    D_fake_prob, D_fake_logit = discriminator(G, reuse=True)


    # Defining Losses
    sig_loss = 0.1*tf.reduce_mean(tf.square(zsig-1))
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit, tf.ones_like(D_logit)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logit, tf.zeros_like(D_fake_logit)))
    gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake_logit, tf.ones_like(D_fake_logit)))
    gloss1 = gloss+sig_loss
    dloss = d_loss_real + d_loss_fake

	
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    data = np.random.normal(0,0.3,(200,2))			# Comment this line when using multimodal (i.e. Uncomment for unimodal data)
    data1 = np.random.normal(0,0.3,(200,2))			# Comment this line when using multimodal (i.e. Uncomment for unimodal data)
    #data = np.random.normal(0.6,0.15,(200,2))			# Uncomment this line for multimodal data
    #data1 = np.random.normal(-0.6,0.15,(200,2))		# Uncomment this line for multimodal data
    data = np.vstack((data,data1))
    data = data.reshape([-1,2])

    # Optimization
    d_optim = tf.train.AdamOptimizer(lr1, beta1=beta1).minimize(dloss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(lr1, beta1=beta1).minimize(gloss1, var_list=g_vars)

    tf.initialize_all_variables().run()
    saver = tf.train.Saver(max_to_keep=10)

    counter = 1
    start_time = time.time()
    data_size = data.shape[0]
    display_z = np.random.normal(0, 1.0, [batchsize, z_dim]).astype(np.float32)				#Uncomment this line for using a mixture of normal prior
    #display_z = np.random.uniform(-1.0, 1.0, [batchsize, z_dim]).astype(np.float32)			#Uncomment this line for using a mixture of uniform distributions prior

    seed = 1
    rng = np.random.RandomState(seed)
    train = True
    thres=1.0
    count=0
    t1=0.73

    for epoch in xrange(8000):
		batch_idx = data_size/batchsize 
		batch = data[rng.permutation(data_size)]
		if count<-1000:
			t1=max(t1-0.005, 0.70)
		lr = learningrate
		for idx in xrange(batch_idx):
			batch_images = batch[idx*batchsize:(idx+1)*batchsize]
			batch_z = np.random.normal(0, 1.0, [batchsize, z_dim]).astype(np.float32)
			batch_z = np.random.uniform(-1.0, 1.0, [batchsize, z_dim]).astype(np.float32)
			
			# Threshold to decide the which phase to run (generator or discrminator phase)
			if count>10:
				thres=min(thres+0.01, 1.0)
				count=0
			if count<-150 and thres>t1:
				thres=max(thres-0.001, t1)
				count=0

			# Training each phase based on the value of thres and gloss
			for k in xrange(5):
				if gloss.eval({z: batch_z})>thres:
					sess.run([g_optim],feed_dict={z: batch_z, lr1:lr})
					count+=1
				else:
					sess.run([d_optim],feed_dict={ images: batch_images, z: batch_z, lr1:lr })
					count-=1

			counter += 1
			# Printing training status periodically
			if counter % 300 == 0:
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, "  % (epoch, idx, batch_idx, time.time() - start_time,))
				sdata = sess.run(G,feed_dict={ z: display_z })
				errD_fake = d_loss_fake.eval({z: display_z})
				errD_real = d_loss_real.eval({images: batch_images})
				errG = gloss.eval({z: display_z})
				sl = sig_loss.eval({z: display_z})
				print('D_real: ', errD_real)
				print('D_fake: ', errD_fake)
				print('G_err: ', errG)
				print('zloss: ', sl)
				 
			# Plotting the generated samples and the training data
			if counter % 1000 == 0:
				f, (ax1,ax2, ax3) = Plot.subplots(1, 3)
				ax1.set_autoscale_on(False)
				ax2.set_autoscale_on(False)
				lab1 = lab.eval({z:display_z})
				gen = G.eval({z:display_z})
				ax1.scatter(gen[:,0], gen[:,1]);
				#ax1.scatter(gen[lab1,0], gen[lab1,1], color='r');			# Uncomment this line when testing with multimodal data
				ax1.set_title('Generated samples')
				ax1.set_aspect('equal')
				ax1.axis([-1,1,-1,1])
				ax2.scatter(batch[:,0], batch[:,1])
				lab_ = batch[batch[:,0]<-0.1]
				#ax2.scatter(lab_[:,0], lab_[:,1], color='r');				# Uncomment this line when testing with multimodal data
				ax2.set_title('Training samples')
				ax2.set_aspect('equal')
				ax2.axis([-1,1,-1,1])
				f.savefig(results_dir + '/plot' + str(counter) + ".png")
				saver.save(sess, os.getcwd()+results_dir+'/train/',global_step=counter)
