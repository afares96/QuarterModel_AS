# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:19:24 2019

@author: Ahmad Fares
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot  as plt
from collections import deque
from tensorflow.python.training import checkpoint_utils as cp
from numpy.random import seed
seed(1234)
from tensorflow import set_random_seed
set_random_seed(1234)
tf.reset_default_graph()

#tf.reset_default_graph() # Reset Tensorflow Graph

# SMD properties
ks = 900
kus = 2500
ms = 2.45
mus = 1
bs = 7.5
bus = 5
A = np.array([[0,1,0,0],[-ks/ms,-bs/ms,ks/ms,bs/ms],[0,0,0,1],[ks/mus,bs/mus,-(ks+kus)/mus,-(bs+bus)/mus]])
B = np.array([[0,0],[0,1/ms,],[-1,0],[bus/mus,-1/mus]])
C = np.array([[1, 0, -1, 0], [-ks/ms, -bs/ms, ks/ms, bs/ms]])
D = np.array([[0], [1/ms]]) 

# Simulation Parameters
dt = 0.01
position = []
actions = np.array([[-0.5],[-0.25],[0],[0.25],[0.5]])
r1 = np.array([[0],[0.2],[0.3],[-0.3],[0.1]])
state_placeholder = tf.placeholder(tf.float32,shape=(None,1))
action_placeholder = tf.placeholder(tf.float32)
delta_placeholder = tf.placeholder(tf.float32)
target_placeholder = tf.placeholder(tf.float32)
stateValue_placeholder = tf.placeholder(tf.float32)
reward_placeholder = tf.placeholder(tf.float32)
gamma_placeholder = tf.placeholder(tf.float32)
nextState_placeholder = tf.placeholder(tf.float32,shape=(None,1))

'''
wa1 = cp.load_variable('neuralWeights1/savenet.ckpt-40','actor/dense/kernel')
wa2 = cp.load_variable('neuralWeights1/savenet.ckpt-40','actor/dense_1/kernel')
wa3 = cp.load_variable('neuralWeights1/savenet.ckpt-40','actor/dense_2/kernel')
wc1 = cp.load_variable('neuralWeights1/savenet.ckpt-40','critic/dense/kernel')
wc2 = cp.load_variable('neuralWeights1/savenet.ckpt-40','critic/dense_1/kernel')
wc3 = cp.load_variable('neuralWeights1/savenet.ckpt-40','critic/dense_2/kernel')
wc4 = cp.load_variable('neuralWeights1/savenet.ckpt-40','critic/dense_3/kernel')
wc5 = cp.load_variable('neuralWeights1/savenet.ckpt-40','critic/dense_4/kernel')
wc6 = cp.load_variable('neuralWeights1/savenet.ckpt-40','critic/dense_5/kernel')
'''
'''
wa11 = wa1[1][:]
wa1 = np.reshape(wa11,(1,5))
'''
'''
wa11 = wa1[1][:]
wa1 = np.reshape(wa11,(1,5))
'''

wa1 = cp.load_variable('nnTest3/savenet.ckpt-118200','actor/dense/kernel')
ba1 = cp.load_variable('nnTest3/savenet.ckpt-118200','actor/dense/bias')
wa2 = cp.load_variable('nnTest3/savenet.ckpt-118200','actor/dense_1/kernel')
ba2 = cp.load_variable('nnTest3/savenet.ckpt-118200','actor/dense_1/bias')
wa3 = cp.load_variable('nnTest3/savenet.ckpt-118200','actor/dense_2/kernel')
ba3 = cp.load_variable('nnTest3/savenet.ckpt-118200','actor/dense_2/bias')
wa4 = cp.load_variable('nnTest3/savenet.ckpt-118200','actor/dense_3/kernel')
ba4 = cp.load_variable('nnTest3/savenet.ckpt-118200','actor/dense_3/bias')
wa5 = cp.load_variable('nnTest3/savenet.ckpt-118200','actor/dense_4/kernel')
ba5 = cp.load_variable('nnTest3/savenet.ckpt-118200','actor/dense_4/bias')
wa6 = cp.load_variable('nnTest3/savenet.ckpt-118200','actor/dense_5/kernel')
ba6 = cp.load_variable('nnTest3/savenet.ckpt-118200','actor/dense_5/bias')

wc1 = cp.load_variable('nnTest3/savenet.ckpt-118200','critic/dense/kernel')
bc1 = cp.load_variable('nnTest3/savenet.ckpt-118200','critic/dense/bias')
wc2 = cp.load_variable('nnTest3/savenet.ckpt-118200','critic/dense_1/kernel')
bc2 = cp.load_variable('nnTest3/savenet.ckpt-118200','critic/dense_1/bias')
wc3 = cp.load_variable('nnTest3/savenet.ckpt-118200','critic/dense_2/kernel')
bc3 = cp.load_variable('nnTest3/savenet.ckpt-118200','critic/dense_2/bias')
wc4 = cp.load_variable('nnTest3/savenet.ckpt-118200','critic/dense_3/kernel')
bc4 = cp.load_variable('nnTest3/savenet.ckpt-118200','critic/dense_3/bias')
wc5 = cp.load_variable('nnTest3/savenet.ckpt-118200','critic/dense_4/kernel')
bc5 = cp.load_variable('nnTest3/savenet.ckpt-118200','critic/dense_4/bias')
wc6 = cp.load_variable('nnTest3/savenet.ckpt-118200','critic/dense_5/kernel')
bc6 = cp.load_variable('nnTest3/savenet.ckpt-118200','critic/dense_5/bias')

wa1i = tf.constant_initializer(wa1)
wa2i = tf.constant_initializer(wa2)
wa3i = tf.constant_initializer(wa3)
wa4i = tf.constant_initializer(wa4)
wa5i = tf.constant_initializer(wa5)
wa6i = tf.constant_initializer(wa6)
ba1i = tf.constant_initializer(ba1)
ba2i = tf.constant_initializer(ba2)
ba3i = tf.constant_initializer(ba3)
ba4i = tf.constant_initializer(ba4)
ba5i = tf.constant_initializer(ba5)
ba6i = tf.constant_initializer(ba6)

wc1i = tf.constant_initializer(wc1)
bc1i = tf.constant_initializer(bc1)
wc2i = tf.constant_initializer(wc2)
bc2i = tf.constant_initializer(bc2)
wc3i = tf.constant_initializer(wc3)
bc3i = tf.constant_initializer(bc3)
wc4i = tf.constant_initializer(wc4)
bc4i = tf.constant_initializer(bc4)
wc5i = tf.constant_initializer(wc5)
bc5i = tf.constant_initializer(bc5)
wc6i = tf.constant_initializer(wc6)
bc6i = tf.constant_initializer(bc6)


# Hyper Parameters, Weights and Biases 
global_step = tf.Variable(0, name='global_step', trainable=False)

def model(x,u,d):  

    xdot = np.matmul(A,x) + B[:,[1]]*u + d
    return (xdot)

def step(x,Torque,d):
    
    k1 = model(x,Torque,d)    
    k2 = model(x + 0.5*k1*dt,Torque,d)
    k3 = model(x + 0.5*k2*dt,Torque,d)
    k4 = model(x + k3*dt,Torque,d)
    x_k = x + (k1 + 2*k2 + 2*k3 + k4)*dt/6
    x_k = x_k.astype(np.float32)
    return (x_k)

def saveParameter(sess,j):
    
    saver = tf.train.Saver() 
    saver.save(sess,"nnTest4\savenet.ckpt",global_step=j)
  
def actor(state):
    n_Hidden = 10
    n_Output = 1  
    with tf.variable_scope("actor", reuse=tf.AUTO_REUSE):
        
  #      hidden1 = tf.layers.dense(state,n_Hidden,tf.nn.elu,use_bias=True,kernel_initializer = wa1i,bias_initializer=ba1i)
        hidden1 = tf.layers.dense(state,n_Hidden,tf.nn.relu,use_bias=True,kernel_initializer = wa1i ,bias_initializer=ba1i)
        hidden2 = tf.layers.dense(hidden1, n_Hidden, tf.nn.tanh,use_bias=True,kernel_initializer = wa2i,bias_initializer=ba2i)
        hidden3 = tf.layers.dense(hidden2, n_Hidden, tf.nn.relu,kernel_initializer = wa3i,bias_initializer=ba3i)
        hidden4 = tf.layers.dense(hidden3, n_Hidden, tf.nn.tanh,kernel_initializer = wa4i,bias_initializer=ba4i) 
        mu  = tf.layers.dense(hidden4, n_Output,None,use_bias=True,kernel_initializer =wa5i ,bias_initializer= ba5i)  
        sigma = tf.layers.dense(hidden4, n_Output,tf.nn.relu,use_bias=True,kernel_initializer = wa6i ,bias_initializer= ba6i)
        sigma = tf.nn.softplus(sigma) 
        norm_dist = tf.contrib.distributions.Normal(mu,sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
       # action_tf_var = tf.clip_by_value(action_tf_var, -15, 15)
    return action_tf_var, norm_dist

def critic(state):
    n_Hidden = 10
    n_Output = 1  
    with tf.variable_scope("critic", reuse= tf.AUTO_REUSE):     
       # hidden1 = tf.layers.dense(state,n_Hidden,tf.nn.relu,use_bias=True,kernel_initializer = wc1i,bias_initializer=bc1i)
       # hidden2 = tf.layers.dense(hidden1, n_Hidden, tf.nn.tanh,use_bias=True,kernel_initializer = wc2i,bias_initializer=bc2i)
       # hidden3 = tf.layers.dense(hidden2, n_Hidden, tf.nn.relu,use_bias=True,kernel_initializer = wc3i,bias_initializer=bc3i)
       # hidden4 = tf.layers.dense(hidden3, n_Hidden, tf.nn.tanh,use_bias=True,kernel_initializer = wc4i,bias_initializer=bc4i)
       # hidden5 = tf.layers.dense(hidden4, n_Hidden, tf.nn.elu,use_bias=True,kernel_initializer = wc5i,bias_initializer=bc5i) 
        hidden1 = tf.layers.dense(state,n_Hidden,tf.nn.relu,use_bias=True,kernel_initializer = wc1i  ,bias_initializer= bc1i)
        hidden2 = tf.layers.dense(hidden1, n_Hidden, tf.nn.tanh,use_bias=True,kernel_initializer = wc2i,bias_initializer= bc2i)
        hidden3 = tf.layers.dense(hidden2, n_Hidden, tf.nn.relu,use_bias=True,kernel_initializer = wc3i,bias_initializer=bc3i)
        hidden4 = tf.layers.dense(hidden3, n_Hidden, tf.nn.tanh,use_bias=True,kernel_initializer = wc4i,bias_initializer=bc4i)
        hidden5 = tf.layers.dense(hidden4, n_Hidden, tf.nn.relu,use_bias=True,kernel_initializer = wc5i,bias_initializer=bc5i) 
        V  = tf.layers.dense(hidden5, n_Output, None,use_bias=True,kernel_initializer = wc6i,bias_initializer=bc6i)  
    return V

'''
def actor(state):
    n_Hidden = 10
    n_Output = 1  
    with tf.variable_scope("actor", reuse=tf.AUTO_REUSE):
        
  #      hidden1 = tf.layers.dense(state,n_Hidden,tf.nn.elu,use_bias=True,kernel_initializer = wa1i,bias_initializer=ba1i)
        hidden1 = tf.layers.dense(state,n_Hidden,tf.nn.relu,use_bias=True,kernel_initializer = tf.contrib.layers.xavier_initializer() ,bias_initializer=tf.zeros_initializer())
        hidden2 = tf.layers.dense(hidden1, n_Hidden, tf.nn.tanh,use_bias=True,kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),bias_initializer=tf.zeros_initializer())
        hidden3 = tf.layers.dense(hidden2, n_Hidden, tf.nn.relu,use_bias=True,kernel_initializer = tf.contrib.layers.xavier_initializer(),bias_initializer=tf.zeros_initializer())
        hidden4 = tf.layers.dense(hidden3, n_Hidden, tf.nn.tanh,use_bias=True,kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),bias_initializer=tf.zeros_initializer()) 
        mu  = tf.layers.dense(hidden4, n_Output,None,use_bias=True,kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev= 0.5, seed=1234) ,bias_initializer=tf.zeros_initializer())  
        sigma = tf.layers.dense(hidden4, n_Output,tf.nn.relu,use_bias=True,kernel_initializer = tf.contrib.layers.xavier_initializer() ,bias_initializer=tf.zeros_initializer())
        sigma = tf.nn.softplus(sigma) 
        norm_dist = tf.contrib.distributions.Normal(mu,sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
       # action_tf_var = tf.clip_by_value(action_tf_var, -15, 15)
    return action_tf_var, norm_dist

def critic(state):
    n_Hidden = 10
    n_Output = 1  
    with tf.variable_scope("critic", reuse= tf.AUTO_REUSE):     
       # hidden1 = tf.layers.dense(state,n_Hidden,tf.nn.relu,use_bias=True,kernel_initializer = wc1i,bias_initializer=bc1i)
       # hidden2 = tf.layers.dense(hidden1, n_Hidden, tf.nn.tanh,use_bias=True,kernel_initializer = wc2i,bias_initializer=bc2i)
       # hidden3 = tf.layers.dense(hidden2, n_Hidden, tf.nn.relu,use_bias=True,kernel_initializer = wc3i,bias_initializer=bc3i)
       # hidden4 = tf.layers.dense(hidden3, n_Hidden, tf.nn.tanh,use_bias=True,kernel_initializer = wc4i,bias_initializer=bc4i)
       # hidden5 = tf.layers.dense(hidden4, n_Hidden, tf.nn.elu,use_bias=True,kernel_initializer = wc5i,bias_initializer=bc5i) 
        hidden1 = tf.layers.dense(state,n_Hidden,tf.nn.relu,use_bias=True,kernel_initializer = tf.contrib.layers.xavier_initializer()  ,bias_initializer=tf.zeros_initializer())
        hidden2 = tf.layers.dense(hidden1, n_Hidden, tf.nn.tanh,use_bias=True,kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),bias_initializer=tf.zeros_initializer())
        hidden3 = tf.layers.dense(hidden2, n_Hidden, tf.nn.relu,use_bias=True,kernel_initializer = tf.contrib.layers.xavier_initializer(),bias_initializer=tf.zeros_initializer())
        hidden4 = tf.layers.dense(hidden3, n_Hidden, tf.nn.sigmoid,use_bias=True,kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),bias_initializer=tf.zeros_initializer())
        hidden5 = tf.layers.dense(hidden4, n_Hidden, tf.nn.relu,use_bias=True,kernel_initializer = tf.contrib.layers.xavier_initializer(),bias_initializer=tf.zeros_initializer()) 
        V  = tf.layers.dense(hidden5, n_Output, None,use_bias=True)  
    return V
'''
class replay_buffer():
   # global buffer
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
        
    def add(self, experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
         buffer_size = len(self.buffer)
         index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
         #rand = randint(0,len(buffer))
         #return buffer[rand]
         return [self.buffer[i] for i in index]


action_tf_var, norm_dist = actor(state_placeholder)
V = critic(state_placeholder)

target = reward_placeholder + gamma_placeholder* critic(nextState_placeholder)
delta = target - V
# Do the target and TD Error 

# define critic (state-value) loss function
loss_critic = tf.reduce_mean(tf.squared_difference(target, V))
training_op_critic = tf.train.AdamOptimizer(0.001, name='critic_optimizer').minimize(loss_critic)
#training_op_critic = tf.compat.v1.train.MomentumOptimizer(0.00001,0.9, name='critic_optimizer').minimize(loss_critic)

# define actor (policy) loss function
loss_actor = -(tf.log(norm_dist.prob(action_placeholder)) * delta)
training_op_actor = tf.train.AdamOptimizer(0.0001, name='actor_optimizer').minimize(loss_actor)
#training_op_actor = tf.compat.v1.train.MomentumOptimizer(0.000001,0.9, name='actor_optimizer').minimize(loss_actor)


with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    zr = 0.0
    #zrlist = np.array([[-0.04],[-0.02],[-0.01],[0],[0.02],[0.04]],dtype=np.float32)
    zrlist = np.linspace(-0.04,0.04,num = 9,dtype = np.float32)
  #  zrdotlist = np.array([[-0.02],[-0.01],[0],[0.02],[0.04]],dtype=np.float32)
    state_current = np.array([[0],[0],[0],[0]],dtype=np.float32)
    ref = np.array([[0],[0],[zr],[0]],dtype=np.float32)
    #saveParameter(sess,6)
    act = []
    smdState = []
    succ = 0
    idx = 0
    lossActor = []
    rf = np.array([[0.02],[0.0]])    
    succ1 = 0
    #memory = replay_buffer(20000)
    a = []
    zr1 = 0.02
    a = 0
    q = 2
    c = []
    zrplot= []
    zrdotplot = []
    j = []
    accel2 = []
    rewards= []
    values = []
    for i in range(40000000):   
        succ1 = succ
        if i%100 == 0:
            print(i)
        '''
        if i % 150 == 0:
            #q = np.random.randint(0,8)
            q = np.random.randint(0,4)/100
            zrold = zr
            #zr = zrlist[q]
            zr = q
            zrdot = (zr-zrold)/dt
            a = 0
            zr1 = zr
            print(zr)
        else:
            zrdot = 0
        '''
        if i % 150 == 0:
            zrold = zr
            if zr == 0:
                zr = 0.02
                zr1 = zr
            else:
                zr = 0
                zr1 = zr
            zrdot = (zr-zrold)/dt
        else:
            zrdot = 0  
        
        if i%600 == 0 :
            plt.plot(rewards)
            plt.show()
            plt.plot(act)
            plt.show()
          #  plt.plot(act)
          #  plt.show()
          #  plt.plot(lossActor)
          #  plt.show()
          #  plt.plot(c)
          #  plt.show()
            plt.plot(j)
            plt.show()
           # plt.plot(zrplot)
           # plt.show()
           # plt.plot(zrdotplot)
           # plt.show()
            plt.plot(smdState)
            plt.show()
            print(succ)
            accelAvg = (np.sum(np.absolute(accel2))/600)
            print(accelAvg)
            saveParameter(sess,i)
            c = []
            lossActor = []
            smdState = []
            act = []
            j = []
            zrplot = []
            zrdotplot = []
            accel2 = []
            rewards= []
            values = []
        xus = state_current[2][0]
        ref = np.array([[xus],[0],[zr],[0]],dtype=np.float32)   
        state = state_current - ref 
        state = state.transpose()
        nnInput = state[0][1]
        nnInput = nnInput.reshape(1,1)
        action = sess.run(action_tf_var, feed_dict = {state_placeholder: nnInput})
        d = np.array([[0],[0],[0],[bus/mus*zrdot+kus/mus*zr1]])
        nextState = step(state_current,action,d)
        xus = nextState[2][0]
        ref = np.array([[xus],[0],[zr],[0]],dtype=np.float32)
        newState = nextState - ref
        newState = newState.transpose()
        stateobs = np.matmul(C,nextState) + D*np.squeeze(action)
        nnInputnew = newState[0][1]
        nnInputnew = nnInputnew.reshape(1,1)
        if np.abs(np.squeeze(stateobs[1][0])) < 0.01:
            #R =  -1000*np.square(np.squeeze(nnInputnew))-0.1*np.abs(np.squeeze(action))
            R = 10
            G = [[0.0]] 
            succ = succ + 1           
        else:
            #R = -100000000*np.square(newState[0][0])
         #   R = 1/(np.square(np.squeeze(newState[0][0]))+0.01)
      #  R = -100*np.square(np.squeeze(nnInputnew)) -0.01*np.square(np.squeeze(action)) - 10*np.square(np.squeeze(stateobs[1][0]))
            R = -1000*np.square(np.squeeze(nnInputnew))-0.1*np.abs(np.squeeze(action))
            G = [[0.99]]  
        smdState.append(state_current[0][0])
        act.append(state[0][0])
        lossActor.append(state_current[2][0])
        c.append(state_current[3][0])
        j.append(np.squeeze(action))
        rewards.append(R)
        zrplot.append(zr)
        zrdotplot.append(zrdot)
        accel2.append(stateobs[1][0])
        state_current = nextState 
       
        #if succ>succ1:
          #  saveParameter(sess,i)
        #VnextState = sess.run(V, feed_dict = {state_placeholder: nnInputnew})
       # target = R + G * np.squeeze(VnextState)
        Val = sess.run(V, feed_dict = {state_placeholder: nnInput})
        values.append(np.squeeze(Val))
        
        #tdError = target - V
        _, loss_critic_val  = sess.run([training_op_critic, loss_critic],feed_dict={state_placeholder: nnInput, nextState_placeholder: nnInputnew, reward_placeholder: R, gamma_placeholder: G})
        _, loss_actor_val  = sess.run([training_op_actor, loss_actor], feed_dict={action_placeholder: np.squeeze(action), state_placeholder: nnInput,nextState_placeholder: nnInputnew, reward_placeholder: R, gamma_placeholder: G})    
        












