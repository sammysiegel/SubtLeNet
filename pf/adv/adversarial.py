import tensorflow as tf
import numpy as np
import utils 
import matplotlib.pyplot as plt
import seaborn as sns # for pretty plots
from scipy.stats import norm

def momentum_optimizer(loss,var_list,train_iters):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.001,batch,train_iters // 4,0.95,staircase=True)
    optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=batch,var_list=var_list)
    #optimizer=tf.train.MomentumOptimizer(learning_rate,0.6).minimize(loss,global_step=batch,var_list=var_list)
    return optimizer

def mlp(input, output_dim):
    # construct learnable parameters within local scope
    w1=tf.get_variable("w0", [input.get_shape()[1], 6], initializer=tf.random_normal_initializer())
    b1=tf.get_variable("b0", [6], initializer=tf.constant_initializer(0.0))
    w2=tf.get_variable("w1", [6, 5], initializer=tf.random_normal_initializer())
    b2=tf.get_variable("b1", [5], initializer=tf.constant_initializer(0.0))
    w3=tf.get_variable("w2", [5,output_dim], initializer=tf.random_normal_initializer())
    b3=tf.get_variable("b2", [output_dim], initializer=tf.constant_initializer(0.0))
    # nn operators
    fc1=tf.nn.tanh(tf.matmul(input,w1)+b1)
    fc2=tf.nn.tanh(tf.matmul(fc1,w2)+b2)
    fc3=tf.nn.tanh(tf.matmul(fc2,w3)+b3)
    return fc3, [w1,b1,w2,b2,w3,b3]

def admlp(train_iters):
    with tf.variable_scope("G"):
        z_node=tf.placeholder(tf.float32, shape=(M,1)) # M uniform01 floats
        G,theta_g=mlp(z_node,1) # generate normal transformation of Z
        G=tf.multiply(5.0,G) # scale up by 5 to match range
        
    with tf.variable_scope("D") as scope:
        x_node=tf.placeholder(tf.float32, shape=(M,1)) # input M normally distributed floats
        fc,theta_d=mlp(x_node,1) # output likelihood of being normally distributed
        D1=tf.maximum(tf.minimum(fc,.99), 0.01) # clamp as a probability
        scope.reuse_variables()
        fc,theta_d=mlp(G,1)
        D2=tf.maximum(tf.minimum(fc,.99), 0.01)
    obj_d=tf.reduce_mean(tf.log(D1)+tf.log(1-D2))
    obj_g=tf.reduce_mean(tf.log(D2))
    opt_d=momentum_optimizer(1-obj_d, theta_d,train_iters)
    opt_g=momentum_optimizer(1-obj_g, theta_g,train_iters) # maximize log(D(G(z)))
    return obj_d,obj_g,theta_d,theta_g,G,D,D2,z_node
    
def plot_d0(D,input_node,xs,mu,sigma):
    f,ax=plt.subplots(1)
    # p_data
    xs=np.linspace(-5,5,1000)
    ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')
    # decision boundary
    r=1000 # resolution (number of points)
    xs=np.linspace(-5,5,r)
    ds=np.zeros((r,1)) # decision surface
    # process multiple points in parallel in a minibatch
    for i in range(r/M):
        x=np.reshape(xs[M*i:M*(i+1)],(M,1))
        ds[M*i:M*(i+1)]=sess.run(D,{input_node: x})

    ax.plot(xs, ds, label='decision boundary')
    ax.set_ylim(0,1.1)
    plt.legend()

def plot_fig(D,G,input_node,xs,mu,sigma,z_node):
    f,ax=plt.subplots(1)
    xs=np.linspace(-5,5,1000)
    ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')
    # decision boundary
    r=1000 # resolution (number of points)
    xs=np.linspace(-5,5,r)
    ds=np.zeros((r,1)) # decision surface
    # process multiple points in parallel in a minibatch
    for i in range(r/M):
        x=np.reshape(xs[M*i:M*(i+1)],(M,1))
        ds[M*i:M*(i+1)]=sess.run(D,{input_node: x})
    ax.plot(xs, ds, label='decision boundary')
    zs=np.linspace(-5,5,r)
    gs=np.zeros((r,1)) # generator function
    for i in range(r/M):
        z=np.reshape(zs[M*i:M*(i+1)],(M,1))
        gs[M*i:M*(i+1)]=sess.run(G,{z_node: z})
    histc, edges = np.histogram(gs, bins = 10)
    ax.plot(np.linspace(-5,5,10), histc/float(r), label='p_g')
    ax.set_ylim(0,1.1)
    plt.legend()
    
TRAIN_ITERS=10000
M=200 # minibatch size
xs=np.linspace(-5,5,1000)
mu,sigma=-1,1

with tf.variable_scope("D_pre"):
    input_node=tf.placeholder(tf.float32, shape=(M,1))
    train_labels=tf.placeholder(tf.float32,shape=(M,1))
    D,theta=mlp(input_node,1)
    loss=tf.reduce_mean(tf.square(D-train_labels))
optimizer=momentum_optimizer(loss,None,TRAIN_ITERS)
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
#plot_d0(D,input_node,xs,mu,sigma)
#plt.title('Initial Decision Boundary')

lh=np.zeros(1000)
for i in range(1000):
    #d=np.random.normal(mu,sigma,M)
    d=(np.random.random(M)-0.5) * 10.0 # instead of sampling only from gaussian, want the domain to be covered as uniformly as possible
    labels=norm.pdf(d,loc=mu,scale=sigma)
    lh[i],_=sess.run([loss,optimizer], {input_node: np.reshape(d,(M,1)), train_labels: np.reshape(labels,(M,1))})

weightsD=sess.run(theta)
sess.close()
obj_g,obj_d,theta_d,theta_g,G,D1,D2,z_node=admlp(TRAIN_ITERS)

for i,v in enumerate(theta_d):
    sess.run(v.assign(weightsD[i]))
plot_fig(D1,G,x_node,xs,mu,sigma,z_node)

k=1
histd, histg= np.zeros(TRAIN_ITERS), np.zeros(TRAIN_ITERS)
for i in range(TRAIN_ITERS):
    for j in range(k):
        x= np.random.normal(mu,sigma,M) # sampled m-batch from p_data
        x.sort()
        z= np.linspace(-5.0,5.0,M)+np.random.random(M)*0.01  # sample m-batch from noise prior
        histd[i],_=sess.run([obj_d,opt_d], {x_node: np.reshape(x,(M,1)), z_node: np.reshape(z,(M,1))})
    z= np.linspace(-5.0,5.0,M)+np.random.random(M)*0.01 # sample noise prior
    histg[i],_=sess.run([obj_g,opt_g], {z_node: np.reshape(z,(M,1))}) # update generator
    if i % (TRAIN_ITERS//10) == 0:
        print(float(i)/float(TRAIN_ITERS))
