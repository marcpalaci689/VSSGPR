'''
@author: Marc Palaci-Olgun
@date: 10/13/2017
@description: The following scripts contains an SGD (stochastic gradient descent) algorithm. It uses RMSPROP
with accelerated Nesterov momentum and a scheduled learning rate. Note that it is not suitable for Big
Data since we still must put the entire data set as an input. However this should not be an issue with datasets
of moderately large size (should easily be able to handle 10^6 data points).

INPUTS:
    objective :  The objective function to be optimized. The objective function should be
                 of the form objective(parameters,x,y) where parameters are the quantities we are optimizing
                 and x and y correspond to the training data.
    parameters : An initial guess for the M parameters we wish to optimize. Should be an array of shape (M,).
    gradients  : The gradients of the objective function with respect to the parameters. Again, the gradients
                 should be of the form gradients(parameters,x,y)
    x          : The entire training data set which should be an array of shape (N,D) where N is the number
                 of training points and D is the dimensionality of the data.
    y          : An (N,1) array of the training targets.
    batch_size : The size of the batches. Initialized to 100.
    step_size  : The step size to use (known as Learning Rate in the Deep Learning community). Initialized to 1e-4.
    epochs     : the maximum number of epochs
    tol        : The minimum allowed norm of the average update over an epoch.
'''

import numpy as np
import math

def RMSPROP(objective,parameters,gradients,x,y,batch_size=100,step_size=1e-3,momentum=0.9,epochs=1000,tol=1e-3,verbose=20):
    exit_flag=False
    
    # record the number of training points
    N = x.shape[0]   
 
    # compute how many complete batches
    batches = int(math.ceil(float(N)/batch_size))   
        
    # compute the decay rate
    decay_rate = step_size/epochs
    
    # initalize Nesterov momentum and RMS gradients to 0
    v = np.zeros(parameters.shape)
    grad_mean_sqr = np.zeros(parameters.shape)
    
    for i in xrange(epochs):
        # compute scheduled learning rate
        step_size *= (1. / (1. + decay_rate * i)) 
        
        # go through entire dataset wih batches
        for j in xrange(batches):
            epoch_update = 0
            if j!= batches-1:
                # compute gradients
                g = gradients(parameters-momentum*v,x[j*batch_size:(j+1)*batch_size],y[j*batch_size:(j+1)*batch_size])
                # compute RMS gradients
                grad_mean_sqr= 0.9*grad_mean_sqr+0.1*g**2
                # compute nesterov accelerated gradients 
                v = momentum*v + step_size*g/np.sqrt(grad_mean_sqr)
                # update parameters 
                parameters -= v
                
                epoch_update += v
                
            # if we are at the last batch we may have to adapt the final batch size to accomodate the data set size
            else:
                # compute RMS gradients
                g = gradients(parameters-momentum*v,x[j*batch_size:],y[j*batch_size:])
                # compute RMS gradients
                grad_mean_sqr= 0.9*grad_mean_sqr+0.1*g**2
                # compute nesterov accelerated gradients (unless its the first iteration)
                v = momentum*v + step_size*g/np.sqrt(grad_mean_sqr)
                # update parameters via the SGD update
                parameters -= v
                epoch_update += v
                
        # shuffle data for the next epoch
        permutation = np.random.permutation(N)
        x = x[permutation]
        y = y[permutation]
        # compute the average update over the last epoch
        epoch_update /= batches
        
        if verbose==-1:
            continue
        else:
            if i%verbose == 0 :
                print objective(parameters,x,y)
       
        # if the average update is smaller than the tolerance then we "converged" 
        if np.linalg.norm(epoch_update) < tol and i>10:
            if not exit_flag:
                exit_flag = True
            else:
                obj = objective(parameters,x,y)
                return {'x':parameters,'nit':i+1,'func':obj}
        else:
            exit_flag = False
            
    obj = objective(parameters,x,y)         
    return {'x':parameters,'nit':i+1,'func':obj}
    