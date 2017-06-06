# This is the code for experiments performed on the CIFAR-10  dataset for the DeLiGAN model. Minor adjustments 
# in the code as suggested in the comments can be done to test GAN. Corresponding details about these experiments 
# can be found in section 5.4 of the paper and the results showing the outputs can be seen in Fig 5 and Table 1.

import argparse
import cPickle
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
import plotting
import cifar10_data
import params

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='../datasets/cifar-10-python')
parser.add_argument('--results_dir', type=str, default='../results/cifar-10-python')
parser.add_argument('--count', type=int, default=400)
args = parser.parse_args()
print(args)

# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# load CIFAR-10 and sample 2000 random images
trainx, trainy = cifar10_data.load(args.data_dir, subset='train')
ind = rng.permutation(trainx.shape[0])
trainx = trainx[ind]
trainy = trainy[ind]
trainx = trainx[:2000]
trainy = trainy[:2000]
trainx_unl = trainx.copy()
testx, testy = cifar10_data.load(args.data_dir, subset='test')
nr_batches_train = int(trainx.shape[0]/args.batch_size)
nr_batches_test = int(testx.shape[0]/args.batch_size)

# specify generative model
noise_dim = (args.batch_size, 100)
Z = th.shared(value=rng.uniform(-1.0,1.0,noise_dim).astype(np.float32), name='Z', borrow=True)
sig = th.shared(value=rng.uniform(-0.2, 0.2,noise_dim).astype(np.float32), name='sig', borrow=True)
noise = theano_rng.normal(size=noise_dim)
#one_hot = T.eye(args.batch_size)  								# Uncomment this line for training/testing MoE-GAN
#noise = T.concatenate([noise, one_hot], axis=1) 						# Uncomment this line for training/testing MoE-GAN
#gen_layers = [ll.InputLayer(shape=(args.batch_size,100 + args.batch_size), input_var=noise)] 	# Uncomment this line for training/testing MoE-GAN
gen_layers = [ll.InputLayer(shape=noise_dim, input_var=noise)]
gen_layers.append(nn.MoGLayer(gen_layers[-1], noise_dim=noise_dim, z=Z,sig=sig))  # Comment this line for training/testing baseline GAN models like GAN, GAN++, MoE-GAN
#gen_layers.append(ll.DenseLayer(gen_layers[-1], num_units=args.batch_size, W=Normal(0.05), nonlinearity=nn.relu)) # Uncomment this line when testing GAN++ 
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (args.batch_size,512,4,4)))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1))
gen_dat = ll.get_output(gen_layers[-1])

# specify discriminative model
disc_layers = [ll.InputLayer(shape=(None, 3, 32, 32))]
disc_layers.append(ll.GaussianNoiseLayer(disc_layers[-1], sigma=0.2))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
disc_layers.append(nn.MinibatchLayer(disc_layers[-1], num_kernels=100))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=10, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))

# costs
labels = T.ivector()
x_lab = T.tensor4()
x_unl = T.tensor4()
temp = ll.get_output(gen_layers[-1], deterministic=False, init=True)
temp = ll.get_output(disc_layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in gen_layers+disc_layers for u in getattr(l,'init_updates',[])]

output_before_softmax_lab = ll.get_output(disc_layers[-1], x_lab, deterministic=False)
output_before_softmax_unl = ll.get_output(disc_layers[-1], x_unl, deterministic=False)
output_before_softmax_gen = ll.get_output(disc_layers[-1], gen_dat, deterministic=False)
sig1 = gen_layers[1].get_sig()     # Comment this line for training/testing baseline GAN models
#sig1 = sig                        # Uncomment this line for training/testing baseline GAN models
sigloss =T.mean((1-sig1)*(1-sig1))*.05
l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels]
l_unl = nn.log_sum_exp(output_before_softmax_unl)
l_gen = nn.log_sum_exp(output_before_softmax_gen)
loss_lab = -T.mean(l_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_lab)))
loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(T.nnet.softplus(l_unl)) + 0.5*T.mean(T.nnet.softplus(l_gen))

train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

# test error
output_before_softmax = ll.get_output(disc_layers[-1], x_lab, deterministic=True)
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))

# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_lab + args.unlabeled_weight*loss_unl, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates)
train_batch_disc = th.function(inputs=[x_lab,labels,x_unl,lr], outputs=[loss_lab, loss_unl, train_err], updates=disc_param_updates+disc_avg_updates)
test_batch = th.function(inputs=[x_lab,labels], outputs=test_err, givens=disc_avg_givens)
samplefun = th.function(inputs=[],outputs=gen_dat)

# Theano functions for training the gen net
loss_gen = -T.mean(T.nnet.softplus(l_gen))
gen_params = ll.get_all_params(gen_layers[-1], trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
train_batch_gen = th.function(inputs=[lr], outputs=[sig1,sigloss,loss_gen], updates=gen_param_updates)

#Uncomment this block for generating GAN samples from given model
'''
f = np.load(args.results_dir + '/disc_params1180.npz')
param_values = [f['arr_%d' % i] for i in range(len(f.files))]
for i,p in enumerate(disc_params):
    p.set_value(param_values[i])
print("disc_params fed")
f = np.load(args.results_dir + '/gen_params1180.npz')
param_values = [f['arr_%d' % i] for i in range(len(f.files))]
for i,p in enumerate(gen_params):
    p.set_value(param_values[i])
print("gen_params fed")
samples=[]
for i in range(50):
    sample_x = samplefun()
    samples.append(sample_x)
samples = np.concatenate(samples,0)
print(samples)
#sys.exit()
np.save(args.results_dir + '/samples50k.npy', samples)
print("samples saved")
sys.exit()
'''


inds = rng.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
#  Uncomment this block when training on the entire dataset
'''
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy==j][:args.count])
    tys.append(trainy[trainy==j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)
'''
a=[]
# //////////// perform training //////////////
for epoch in range(1200):
    begin = time.time()
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3. - epoch/400., 1.))
    # Uncomment this block when training on the entire dataset
    '''
    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    for t in range(int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0])))):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    '''
    if epoch==0:
        init_param(trainx[:500]) # data based initialization

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    for t in range(nr_batches_train):
        ll, lu, te = train_batch_disc(trainx[t*args.batch_size:(t+1)*args.batch_size],trainy[t*args.batch_size:(t+1)*args.batch_size],
                                        trainx_unl[t*args.batch_size:(t+1)*args.batch_size],lr)
        loss_lab += ll
        loss_unl += lu
        train_err += te
        
        for rep in range(3):
            sigm, sigmloss, genloss = train_batch_gen(lr)

    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train
    
    # test
    test_err = 0.
    for t in range(nr_batches_test):
        test_err += test_batch(testx[t*args.batch_size:(t+1)*args.batch_size],testy[t*args.batch_size:(t+1)*args.batch_size])
    test_err /= nr_batches_test

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err= %.4f, test err = %.4f, gen_loss = %.4f, sigloss = %.4f" %(epoch, time.time()-begin, loss_lab, loss_unl,train_err,test_err,genloss,sigmloss))
    sys.stdout.flush()
    a.append([epoch, loss_lab, loss_unl, train_err, test_err,genloss,sigmloss])
    # generate samples from the model
    sample_x = samplefun()
    img_bhwc = np.transpose(sample_x[:100,], (0, 2, 3, 1))
    img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title='CIFAR10 samples')
    plotting.plt.savefig(args.results_dir + '/dg_cifar10_sample_minibatch.png')
    if epoch%20==0:
        NNdiff = np.sum(np.sum(np.sum(np.square(np.expand_dims(sample_x,axis=1)-np.expand_dims(trainx,axis=0)),axis=2),axis=2),axis=2)
        NN = trainx[np.argmin(NNdiff,axis=1)]
        NN = np.transpose(NN[:100], (0, 2, 3, 1))
        NN_tile = plotting.img_tile(NN, aspect_ratio=1.0,border_color=1.0,stretch=True)
        img_tile = np.concatenate((img_tile,NN_tile),axis=1)
        img = plotting.plot_img(img_tile, title='CIFAR10 samples')
        plotting.plt.savefig(args.results_dir + '/'+str(epoch)+'.png')
        # save params
        np.savez(args.results_dir + '/disc_params'+str(epoch)+'.npz',*[p.get_value() for p in disc_params])
        np.savez(args.results_dir + '/gen_params'+str(epoch)+'.npz',*[p.get_value() for p in gen_params])
        np.save(args.results_dir + '/train/errors.npy',a)
        np.save(args.results_dir + '/train/sig.npy',sigm)
    plotting.plt.close('all')


