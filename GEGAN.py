

# It is not stable now, the lose of GAN may be nan in the initializtion. You can try more times to have good result. 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mb_size = 128
X_dim = 784


hd_dim = 512
hd_dim1 = 256 
hd_dim2 = 128
hd_dim3 = 32


z_dim = 10
h_dim = 16
h1_dim= 256
h2_dim= 512


lam = 10
n_disc = 15
n_gene = 10

lr = 1e-4


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Good D

X = tf.placeholder(tf.float32, shape=[None, X_dim]) #[,784] array

D_W1 = tf.Variable(xavier_init([X_dim, hd_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[hd_dim]))

D_WM1 = tf.Variable(xavier_init([hd_dim, hd_dim1]))
D_bM1 = tf.Variable(tf.zeros(shape=[hd_dim1]))

D_WM2 = tf.Variable(xavier_init([hd_dim1, hd_dim2]))
D_bM2 = tf.Variable(tf.zeros(shape=[hd_dim2]))

D_WM3 = tf.Variable(xavier_init([hd_dim2, hd_dim3]))
D_bM3 = tf.Variable(tf.zeros(shape=[hd_dim3]))

D_W2 = tf.Variable(xavier_init([hd_dim3, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))


theta_D = [D_W1,D_WM1,D_WM2,D_WM3 , D_W2, D_b1,D_bM1,D_bM2,D_bM3, D_b2]
initail_D = tf.variables_initializer(theta_D)

# Evil D

ED_W1 = tf.Variable(xavier_init([X_dim, hd_dim]))
ED_b1 = tf.Variable(tf.zeros(shape=[hd_dim]))

ED_WM1 = tf.Variable(xavier_init([hd_dim, hd_dim1]))
ED_bM1 = tf.Variable(tf.zeros(shape=[hd_dim1]))

ED_WM2 = tf.Variable(xavier_init([hd_dim1, hd_dim2]))
ED_bM2 = tf.Variable(tf.zeros(shape=[hd_dim2]))

ED_WM3 = tf.Variable(xavier_init([hd_dim2, hd_dim3]))
ED_bM3 = tf.Variable(tf.zeros(shape=[hd_dim3]))

ED_W2 = tf.Variable(xavier_init([hd_dim3, 1]))
ED_b2 = tf.Variable(tf.zeros(shape=[1]))


theta_ED = [ED_W1,ED_WM1,ED_WM2,ED_WM3 , ED_W2, ED_b1,ED_bM1,ED_bM2,ED_bM3, ED_b2]
initail_ED = tf.variables_initializer(theta_ED)

# G

z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_WM1 = tf.Variable(xavier_init([h_dim, h1_dim]))
G_bM1 = tf.Variable(tf.zeros(shape=[h1_dim]))

G_WM2 = tf.Variable(xavier_init([h1_dim, h2_dim]))
G_bM2 = tf.Variable(tf.zeros(shape=[h2_dim]))

G_W2 = tf.Variable(xavier_init([h2_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_WM1, G_WM2, G_W2, G_b1, G_bM1,G_bM2, G_b2]
initail_G = tf.variables_initializer(theta_G)



def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])



def G(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_WM1) + G_bM1)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_WM2) + G_bM2)
    G_log_prob = tf.matmul(G_h3, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def D(X):
    D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_WM1) + D_bM1)
    D_h3 = tf.nn.relu(tf.matmul(D_h2, D_WM2) + D_bM2)
    D_h4 = tf.nn.relu(tf.matmul(D_h3, D_WM3) + D_bM3)
    out =  tf.nn.sigmoid(tf.matmul(D_h4 , D_W2) + D_b2)
    return out
    
def ED(X):
    ED_h1 = tf.nn.relu(tf.matmul(X, ED_W1) + ED_b1)
    ED_h2 = tf.nn.relu(tf.matmul(ED_h1, ED_WM1) + ED_bM1)
    ED_h3 = tf.nn.relu(tf.matmul(ED_h2, ED_WM2) + ED_bM2)
    ED_h4 = tf.nn.relu(tf.matmul(ED_h3, ED_WM3) + ED_bM3)
    Eout =  tf.nn.sigmoid(tf.matmul(ED_h4 , ED_W2) + ED_b2)
    return Eout



G_sample = G(z)
D_real = D(X)
D_fake = D(G_sample)
ED_real = ED(G_sample)
ED_fake = ED(X)


#GD
#Improved WGAN parameter
eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
grad = tf.gradients(D(X_inter), [X_inter])[0] 
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean(grad_norm - 0.1)**2


loss0 = -( ED_real*tf.log(D_fake)+(1-ED_real)*tf.log(1-D_fake) )
loss1 = -( 1*tf.log(D_real) )
D_loss = tf.reduce_mean(loss0)+tf.reduce_mean(loss1) + grad_pen
D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(D_loss, var_list=theta_D))

#ED
#Improved WGAN parameter
EX_inter = eps*G_sample + (1. - eps)*X
Egrad = tf.gradients(ED(EX_inter), [EX_inter])[0] 
Egrad_norm = tf.sqrt(tf.reduce_sum((Egrad)**2, axis=1))
Egrad_pen = lam * tf.reduce_mean(Egrad_norm - 0.1)**2

ECrossEntropy = -((D_fake)*tf.log(ED_real)+(1-D_fake)*tf.log(1-ED_real))
ECrossEntropyReal = -( 1*tf.log(1-ED_fake) )
ED_loss =  tf.reduce_mean(ECrossEntropy)+tf.reduce_mean(ECrossEntropyReal)+Egrad_pen
ED_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(ED_loss , var_list=theta_ED))

#G
loss2 = -( 1*tf.log(D_fake)+1*tf.log(1-ED_real) )
G_loss = tf.reduce_mean(loss2)
G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(G_loss, var_list=theta_G))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('GEGANout/'):
    os.makedirs('GEGANout/')

i = 0

for it in range(200001):
    
    for _ in range(n_disc):
        X_mb, _ = mnist.train.next_batch(mb_size)

        sess.run([D_solver,ED_solver] , feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)})
    for _ in range(n_gene):
        sess.run(G_solver , feed_dict={z: sample_z(mb_size, z_dim)} )

    

    if it % 100 == 0:
        D_loss_curr,ED_loss_curr = sess.run([D_loss,ED_loss],feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)})
        G_loss_curr = sess.run(G_loss,feed_dict={z: sample_z(mb_size, z_dim)})
        print('Iter: {}; D loss: {:.4}; ED loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr,ED_loss_curr, G_loss_curr))
       
              
        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={z: sample_z(25, z_dim)})

            fig = plot(samples)
            plt.savefig('GEGANout/{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)