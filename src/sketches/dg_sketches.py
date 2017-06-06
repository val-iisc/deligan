# This is the code for experiments performed on the Eitz Sketches dataset for the DeLiGAN model. Minor adjustments 
# in the code as suggested in the comments can be done to test GAN. Corresponding details about these experiments 
# can be found in section 5.5 of the paper and the results showing the outputs can be seen in Fig 6 and Table 2,3.

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
import input_data_gan

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--data_dir', type=str, default='../datasets/sketches/')
parser.add_argument('--results_dir', type=str, default='../results/sketches/')
parser.add_argument('--count', type=int, default=400)
args = parser.parse_args()
gen_dim = 40 
disc_dim = 20
print(args)

# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# load sketches 
data = input_data_gan.read_data_sets(args.data_dir,one_hot=True).train
trainx = data._images
print("trainx_Shape",trainx.shape)
trainx = trainx.reshape([-1,1,32,32])
trainx = trainx*2.-1
ind = rng.permutation(trainx.shape[0])
trainx = trainx[ind]
nr_batches_train = int(trainx.shape[0]/args.batch_size)

# specify generative model
noise_dim = (args.batch_size, 100)
Z = th.shared(value=rng.uniform(-1.0,1.0,noise_dim).astype(np.float32), name='Z', borrow=True)
sig = th.shared(value=rng.uniform(0.2, 0.2,noise_dim).astype(np.float32), name='sig', borrow=True)
noise = theano_rng.normal(size=noise_dim)
gen_layers = [ll.InputLayer(shape=noise_dim, input_var=noise)]
gen_layers.append(nn.MoGLayer(gen_layers[-1], noise_dim=noise_dim, z=Z, sig=sig))   #  Comment this line when testing/training baseline GAN model
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*gen_dim*4, W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (args.batch_size,gen_dim*4,4,4)))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,gen_dim*2,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4 -> 8
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,gen_dim,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 8 -> 16
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,1,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 16 -> 32
gen_dat = ll.get_output(gen_layers[-1])

# specify discriminative model
disc_layers = [ll.InputLayer(shape=(None, 1, 32, 32))]
disc_layers.append(ll.GaussianNoiseLayer(disc_layers[-1], sigma=0.2))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1],disc_dim,(5,5), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], disc_dim, (5,5), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1],disc_dim*2,(5,5), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1],disc_dim*4,(3,3),pad=0, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=disc_dim*4, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
disc_layers.append(nn.MinibatchLayer(disc_layers[-1], num_kernels=50))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=1, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))

# costs
x_lab = T.tensor4()
temp = ll.get_output(gen_layers[-1], deterministic=False, init=True)
temp = ll.get_output(disc_layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in gen_layers+disc_layers for u in getattr(l,'init_updates',[])]

D_logit = ll.get_output(disc_layers[-1], x_lab, deterministic=False)
D_prob = T.nnet.sigmoid(D_logit)
D_fake_logit = ll.get_output(disc_layers[-1], gen_dat, deterministic=False)
D_fake_prob = T.nnet.sigmoid(D_fake_logit)


sig1 = gen_layers[1].get_sig()        		#  Comment this line when training/testing the baseline GAN Model
# sigma regularizer
sigloss =T.mean((1-sig1)*(1-sig1))*.05     	#  Comment this line when training/testing the baseline GAN Model

#sigloss = th.shared(value=rng.uniform(0,0), name='sigloss', borrow=True)    #  Uncomment this line when training/testing the baseline GAN Model
#sig1 = th.shared(value=rng.uniform(0.2,0.2,noise_dim), name='sig1', borrow=True)    #  Uncomment this line when training/testing the baseline GAN Model
loss_real = T.mean(T.nnet.binary_crossentropy(D_prob,T.ones_like(D_prob)))
loss_fake = T.mean(T.nnet.binary_crossentropy(D_fake_prob,T.zeros_like(D_fake_prob)))

# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_real + args.unlabeled_weight*loss_fake, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates)
train_batch_disc = th.function(inputs=[x_lab,lr], outputs=[loss_real,loss_fake], updates=disc_param_updates+disc_avg_updates)
samplefun = th.function(inputs=[],outputs=gen_dat)

# Theano functions for training the gen net
loss_gen = T.mean(T.nnet.binary_crossentropy(D_fake_prob,T.ones_like(D_fake_prob)))
gen_params = ll.get_all_params(gen_layers[-1],trainable=True)

gen_param_updates = nn.adam_updates(gen_params, loss_gen + sigloss, lr=lr, mom1=0.5)
train_batch_gen = th.function(inputs=[lr], outputs=[sig1,sigloss,loss_gen], updates=gen_param_updates)
batch_gen = th.function(inputs=[], outputs=[sig1,sigloss,loss_gen],updates=None)


# Uncomment this block when generative samples from a pretrained model
'''
f = np.load(args.results_dir + '/train/disc_params3850.npz')
param_values = [f['arr_%d' % i] for i in range(len(f.files))]
for i,p in enumerate(disc_params):
    p.set_value(param_values[i])
print("disc_params fed")
f =np.load(args.results_dir + '/train/gen_params3850.npz')
param_values = [f['arr_%d' % i] for i in range(len(f.files))]
for i,p in enumerate(gen_params):
    p.set_value(param_values[i])
print("gen_params fed")
samples=[]
for i in range(500):
    sample_x = samplefun()
    samples.append(sample_x)
samples = np.concatenate(samples,0)
print(samples)
#sys.exit()
np.save(args.results_dir + '/DE_samples50k.npy',samples)
print("samples saved")
sys.exit()
'''

# select labeled data
inds = rng.permutation(trainx.shape[0])
trainx = trainx[inds]
a = []
count1=0
count2=0
t1=0.70
thres=1.0
# //////////// perform training //////////////
for epoch in range(3900):
    begin = time.time()
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3.-epoch/1300., 1.))
    lrd = np.cast[th.config.floatX](args.learning_rate*0.5* np.minimum(3. - epoch/1300., 1.))

    if epoch==0:
        init_param(trainx[:400]) # data based initialization

    loss_lab = 0
    loss_unl = 0
    train_err = 0

    # train
    for t in range(nr_batches_train):
        sigm , sigmloss, genloss = batch_gen()
        if count1>5:
            thres=min(thres+0.003,1.0)
            count1=0
            ll, lu =train_batch_disc(trainx[t*args.batch_size:(t+1)*args.batch_size],lrd)
            #print('gen',thres)
        if count2<-1:
            thres=max(thres-0.003, t1)
            count2=0
            #print('disc', thres)

        for k in xrange(5):
            if(genloss>thres):
                sigm, sigmloss, genloss = train_batch_gen(lr)
                count1+=1
                count2=0
            else:
                ll, lu = train_batch_disc(trainx[t*args.batch_size:(t+1)*args.batch_size],lr)
                sigm, sigmloss, genloss = batch_gen()
                count1=0
                count2-=1
                loss_lab = ll
                loss_unl = lu


    print("Iteration %d, time = %ds, loss_real = %.4f, loss_fake = %.4f,loss_gen= = %.4f, sigloss = %.4f" %
          (epoch, time.time()-begin, loss_lab, loss_unl,genloss,sigmloss))
    sys.stdout.flush()
    a.append([epoch, loss_lab, loss_unl, genloss, sigmloss])

    # generate samples from the model
    sample_x = samplefun()
    img_bhwc = np.transpose(sample_x[:100,], (0, 2, 3, 1))
    NNdiff = np.sum(np.sum(np.sum(np.square(np.expand_dims(sample_x,axis=1)-np.expand_dims(trainx,axis=0)),axis=2),axis=2),axis=2)
    NN = trainx[np.argmin(NNdiff,axis=1)]
    NN = np.transpose(NN[:100], (0, 2, 3, 1))
    img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
    NN_tile = plotting.img_tile(NN, aspect_ratio=1.0, border_color=1.0,stretch=True)
    img_tile = np.concatenate((img_tile,NN_tile),axis=1)
    img_tile = img_tile.reshape(img_tile.shape[0],img_tile.shape[1])
    img = plotting.plot_img(img_tile, title='sketch samples')
    plotting.plt.savefig(args.results_dir + '/bg_sketch_sample_minibatch.png')
    if epoch%50==0:
        plotting.plt.savefig(args.results_dir + '/'+str(epoch)+'.png')
        # save params
        np.savez(args.results_dir + '/train/disc_params' + str(epoch) + '.npz',*[p.get_value() for p in disc_params])
        np.savez(args.results_dir + '/train/gen_params'+ str(epoch) + '.npz',*[p.get_value() for p in gen_params])
        np.save(args.results_dir + '/train/errors.npy',a)
        np.save(args.results_dir + '/train/sig.npy',sigm)

    plotting.plt.close('all')

