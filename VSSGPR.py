import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
import SGD
import test_functions as test

'''
The following object is a Random Fourier Feature approximation to the Spectral 
Mixture Kernel
'''

class SM:
    def __init__(self,dimensions,features,components):
        self.D = dimensions
        self.M = features
        self.L = components
        # initialize frequencies
        self.w = np.random.normal(size=(self.L,self.M,self.D))
        # initialize lengthscales
        self.l = 1.0*np.ones((self.L,self.D)) 
        # initialize periods
        self.p = 1.0*np.ones((self.L,self.D))
        # initialize amplitude
        self.sigma = 1.0*np.ones(self.L)
        # compute a commonly used scalar
        self.factor = np.sqrt(2.0/self.M)
        
        return
        
    def design_matrix(self,x):
        N = x.shape[0]
        phi = np.zeros((N,self.L*self.M))
        for i in xrange(self.L):
            phi[:,i*self.M:(i+1)*self.M] = self.sigma[i]*np.cos(np.dot(x,\
                ((1.0/self.l[i])*self.w[i]+2*np.pi*self.p[i]).T))
        return self.factor*phi

    def update_lengthscales(self,l):
        self.l = l.reshape(self.L,self.D)
        return
    
    def update_periods(self,p):
        self.p = p.reshape(self.L,self.D)
        return
    
    def update_amplitude(self,sigma):
        self.sigma = sigma
        return
    
    def update_frequencies(self,w):
        self.w = w.reshape(self.L,self.M,self.D)
        return
    

    
class VD:
    def __init__(self,dimensions,features,components,type):
        self.D     = dimensions
        self.M     = features
        self.L     = components
        self.mu    = np.random.normal(size=(self.L,self.M,self.D))
        self.var   = np.random.normal(size=(self.L,self.M,self.D))
        if type in ['full','stochastic']:
            self.m     = np.random.normal(size=(self.L*self.M))
            self.s     = np.random.normal(size=self.L*self.M)
        self.alpha = 2*np.pi*np.random.rand(self.L,self.M)
        
    def update_mu(self,mu):
        self.mu = mu.reshape(self.L,self.M,self.D)
        return
    
    def update_var(self,var):
        self.var = var.reshape(self.L,self.M,self.D)
        return
        
    def update_m(self,m):
        self.m = m
        return
    
    def update_s(self,s):
        self.s = s
        return
         

'''
The following object is a Variational Distribution of the spectral frequencies as well as the
fourier coefficients
'''        
    
class VSSGPR:
    def __init__(self,x_train,y_train,cov,type='full'):
        assert cov.D == x_train.shape[1], 'covariance dimensionality does not match data dimensionality'
        assert type in ['full','exact','stochastic','fast'] , 'type of vssgpr must either be \'full\',\'exact\', or \'fast\' or \'stochastic\' ' 
        self.type       = type
        self.cov        = cov
        self.vd         = VD(cov.D,cov.M,cov.L,self.type)
        self.x_train    = x_train
        self.y_train    = y_train
        self.precision  = 1000.                                # spherical noise parameter
        self.N,self.D   = x_train.shape 
        self.group_parameters()
        self.num_params = len(self.params)
        self.index1     = self.cov.D*self.cov.M*self.cov.L
        self.index2     = self.cov.D*self.cov.L
        return

    def group_parameters(self):
        if self.type in ['full','stochastic']:
            self.params=np.hstack((self.vd.mu.flatten(),self.vd.var.flatten(),self.vd.m.flatten(), \
                               self.vd.s.flatten(),self.cov.sigma,self.cov.l.flatten(), \
                               self.cov.p.flatten()))
        else:
            self.params=np.hstack((self.vd.mu.flatten(),self.vd.var.flatten(),self.cov.sigma,self.cov.l.flatten(), \
                               self.cov.p.flatten()))
        return
    
    def update_parameters(self,params):
        if self.type in ['full','stochastic']:
            self.vd.update_mu(params[:self.index1])
            self.vd.update_var(params[self.index1:2*self.index1])
            self.vd.update_m(params[2*self.index1:3*self.index1])
            self.vd.update_s(params[3*self.index1:4*self.index1])
            self.cov.update_amplitude(params[4*self.index1:4*self.index1+self.cov.L])
            self.cov.update_lengthscales(params[4*self.index1+self.cov.L:4*self.index1+self.cov.L+self.index2])
            self.cov.update_periods(params[4*self.index1+self.cov.L+self.index2:])
        else:
            self.vd.update_mu(params[:self.index1])
            self.vd.update_var(params[self.index1:2*self.index1])
            self.cov.update_amplitude(params[2*self.index1:2*self.index1+self.cov.L])
            self.cov.update_lengthscales(params[2*self.index1+self.cov.L:2*self.index1+self.cov.L+self.index2])
            self.cov.update_periods(params[2*self.index1+self.cov.L+self.index2:])
        self.params = params
        return


    def expectations(self,x):
        Ephi = []
        diag_correction = []
        P     = self.cov.p**-1
        L      = (2*np.pi*self.cov.l)**-1
        x_bar  = 2*np.pi*x
        factor = 2*(self.cov.sigma**2)/self.cov.M
        
        for i in xrange(self.cov.L):
            exp_factor = np.sqrt(factor[i])*np.exp(-0.5*np.dot((L[i]*x_bar)**2,\
                                 (np.exp(self.vd.var[i]).T)))
            eq_factor2 = self.vd.alpha[i] + (np.dot(L[i]*x_bar,self.vd.mu[i].T)\
                               + P[i]*x_bar)
            eq_factor  = np.cos(eq_factor2)
            Ephi.append(exp_factor*eq_factor)
            eq_factor2= np.cos(2*eq_factor2)
            exp_factor = (exp_factor*factor[i]**(-0.5))**4
            diag_correction.append(factor[i]*((0.5+0.5*exp_factor*eq_factor2).sum(0)))
        
        Ephi = np.concatenate(Ephi,axis=1)
        diag_correction = np.concatenate(diag_correction)
        Ephi_squared = np.dot(Ephi.T,Ephi)
        Ephi_squared = Ephi_squared + np.diag(diag_correction-np.diag(Ephi_squared))       
        
        return Ephi, Ephi_squared       
    
    def predict(self,x):
        Ephi_s = []
        diag_correction = []
        P     = self.cov.p**-1
        L      = (2*np.pi*self.cov.l)**-1
        x_bar  = 2*np.pi*x
        factor = 2*(self.cov.sigma**2)/self.cov.M
        
        for i in xrange(self.cov.L):
            exp_factor = np.sqrt(factor[i])*np.exp(-0.5*np.dot((L[i]*x_bar)**2,\
                                 (np.exp(self.vd.var[i]).T)))
            eq_factor2 = self.vd.alpha[i] + (np.dot(L[i]*x_bar,self.vd.mu[i].T)\
                               + P[i]*x_bar)
            eq_factor  = np.cos(eq_factor2)
            Ephi_s.append(exp_factor*eq_factor)
            eq_factor2= np.cos(2*eq_factor2)
            exp_factor = (exp_factor*factor[i]**(-0.5))**4
            diag_correction.append(factor[i]*((0.5+0.5*exp_factor*eq_factor2)))
        
        Ephi_s = np.concatenate(Ephi_s,axis=1)
        diag_correction = np.concatenate(diag_correction,axis=1)
        mu = np.dot(Ephi_s,self.vd.m.reshape(-1,1))
        var = np.zeros((x.shape[0],1))
        
        if self.type in ['full','stochastic']:
            s = np.exp(self.vd.s)
            for i in xrange(x.shape[0]):
                psi = np.sum(diag_correction[i].T*s)
                var[i]= (self.precision**-1.0) + psi + np.sum(self.vd.m**2*(diag_correction[i]-Ephi_s[i]**2))                
        else:
            self.vd.update_s(self.precision**-1*np.linalg.inv(self.sigma))
            s = self.vd.s
            for i in xrange(x.shape[0]):        
                ephi_squared = np.dot(Ephi_s[i].reshape(-1,1),Ephi_s[i].reshape(1,-1)) 
                ephi_squared = ephi_squared + np.diag(diag_correction[i]-np.diag(ephi_squared))
                psi = np.sum(ephi_squared.T*s)
                var[i]= (self.precision**-1.0) + psi + np.sum(self.vd.m.flatten()**2*(diag_correction[i]-Ephi_s[i]**2))
                
        return mu,np.sqrt(var)     
    
    def ELBO(self,params,x,y):
        # record the batch size
        batch_size = x.shape[0]
        
        # Update the parameters
        self.update_parameters(params)
        
        # compute KL divergence of W   
        KL_W = np.sum(0.5*(np.exp(self.vd.var) + self.vd.mu**2 - self.vd.var - 1))
        
        # compute part of the ELBO term for the full VSSGPR
        if self.type in ['full','stochastic']:
            # get the expectations
            Ephi,Ephi_squared = self.expectations(x)
            # compute KL divergence of A  
            KL_A = np.sum(0.5*(np.exp(self.vd.s) + self.vd.m**2 - self.vd.s - 1))
            # compute full elbo
            '''
            elbo = ((float(self.N)/batch_size)*(0.5*batch_size*np.log(2*np.pi*self.precision**-1)\
                   +0.5*self.precision*np.sum(y**2)\
                   -self.precision*(y.T.dot(Ephi).dot(self.vd.m.reshape(-1,1))).sum()\
                   +0.5*self.precision*np.trace(np.dot(Ephi_squared,np.diag(np.exp(self.vd.s))\
                   +np.dot(self.vd.m.reshape(-1,1),self.vd.m.reshape(1,-1)))))\
                    +KL_A+KL_W).sum()    
            '''
            elbo = (float(self.N)/batch_size)*(0.5*batch_size*np.log(2*np.pi*self.precision**-1)\
                    +0.5*self.precision*np.sum(y**2)\
                    -self.precision*np.sum(y*np.dot(Ephi,self.vd.m.reshape(-1,1)))\
                    +0.5*self.precision*np.sum(np.diag(Ephi_squared)*np.exp(self.vd.s))\
                    +0.5*self.precision*np.trace(np.dot(Ephi_squared,np.dot(self.vd.m.reshape(-1,1),self.vd.m.reshape(1,-1)))))\
                    +KL_A+KL_W 
                                                       
        # TODO: FIND MORE EFFICIENT ELBO... why did i do np.dot(Ephi_squared,np.diag(np.exp(self.vd.s))
        else:
            # get the expectations
            Ephi,Ephi_squared = self.expectations(x)
            self.sigma = Ephi_squared + self.precision**-1*np.eye(self.cov.L*self.cov.M)
            self.chol_sigma = np.linalg.cholesky(self.sigma)
            self.vd.update_m(np.linalg.solve(self.chol_sigma.T,np.linalg.solve(self.chol_sigma,np.dot(Ephi.T,y))))
            elbo = ((float(self.N)/batch_size)*(0.5*batch_size*np.log(2*np.pi*self.precision**-1)\
                   +0.5*self.precision*np.sum(y**2)\
                   +0.5*self.cov.L*self.cov.M*np.log(self.precision)\
                   +np.sum(np.log(np.diag(self.chol_sigma))) \
                   -0.5*self.precision*np.dot(y.T,np.dot(Ephi,self.vd.m)).sum())\
                   +KL_W).sum()
           
        return elbo
    
    def ELBO_by_batch(self,params,x,y,batch_size=100):
        assert self.type in ['full','stochastic'] , 'Computing the elbo by batch operations only works using full or stochastic formulations'
        elbo = 0
        n = 0
        # Update the parameters
        self.update_parameters(params)
        
        # compute KL divergence of W   
        KL_W = np.sum(0.5*(np.exp(self.vd.var) + self.vd.mu**2 - self.vd.var - 1))
        KL_A = np.sum(0.5*(np.exp(self.vd.s) + self.vd.m**2 - self.vd.s - 1))
        
        # compute mean covariance 
        M = np.dot(self.vd.m.reshape(-1,1),self.vd.m.reshape(1,-1))
        
        while n <= self.N-batch_size:
            # get the expectations
            Ephi,Ephi_squared = self.expectations(x[n:n+batch_size])
            elbo += 0.5*batch_size*np.log(2*np.pi*self.precision**-1)\
                    +0.5*self.precision*np.sum(y[n:n+batch_size]**2)\
                    -self.precision*np.sum(y[n:n+batch_size]*np.dot(Ephi,self.vd.m.reshape(-1,1)))\
                    +0.5*self.precision*np.sum(np.diag(Ephi_squared)*np.exp(self.vd.s))\
                    +0.5*self.precision*np.trace(np.dot(Ephi_squared,M))
            n += batch_size
            
        if n<self.N:
            batch_size = self.N-n
            Ephi,Ephi_squared = self.expectations(x[n:n+batch_size])
            # get the expectations
            Ephi,Ephi_squared = self.expectations(x[n:])
            elbo += 0.5*batch_size*np.log(2*np.pi*self.precision**-1)\
                    +0.5*self.precision*np.sum(y[n:]**2)\
                    -self.precision*np.sum(y[n:]*np.dot(Ephi,self.vd.m.reshape(-1,1)))\
                    +0.5*self.precision*np.sum(np.diag(Ephi_squared)*np.exp(self.vd.s))\
                    +0.5*self.precision*np.trace(np.dot(Ephi_squared,M))
                       
        return elbo+KL_W+KL_A

    def optimize(self,method='L-BFGS-B',restarts=1,iterations=1000,verbose=True):
        if verbose:
            print('***************************************************')
            print('*              Optimizing parameters              *')
            print('***************************************************')
        
        global_opt = np.inf                                                # Initialize the global optimum value
        # Get gradient function using Autograd 
        gradients = grad(self.ELBO)
        for res in xrange(restarts):
           # Optimize
           if self.type in ['fast','stochastic']:
               self.opt = SGD.RMSPROP(self.ELBO,self.params,gradients,self.x_train,self.y_train,batch_size=100,momentum=0.9,step_size=0.005,epochs=iterations)
           else:
               self.opt = minimize(self.ELBO,self.params,args=(self.x_train,self.y_train),jac=gradients,options={'maxiter':iterations})
        return self.opt
   
    # plot training data along with GP     
    def plot(self):
        # ensure data is only 1D
        assert self.D == 1, 'Plot can only work for 1D data'
        
        # set up the range of the plot
        x_max   = np.max(self.x_train)
        x_min   = np.min(self.x_train)
        padding = (x_max-x_min)*0.15
        
        x = np.linspace(x_min-padding,x_max+padding,1000).reshape(-1,1)
        
        # get GP mean and standard deviations over the range
        mu,std = self.predict(x)
        
        # plot training points along with the GP 
        plt.figure()
        plt.clf()
        plt.plot(x,mu,'g',label='GP Mean')
        plt.fill_between(x.flatten(),(mu+2*std).flatten(),(mu-2*std).flatten(),interpolate=True,alpha=0.25,color='orange',label='95% Confidence')
        plt.scatter(self.x_train,self.y_train,label='Training Points')
        plt.xlabel('X',fontsize=20)
        plt.ylabel('Y',fontsize=20)
        plt.legend()
        plt.show()
        return
        

if __name__ == '__main__':
    x,y = test.step(500)
    cov = SM(1,60,2)
    
    vssgpr = VSSGPR(x,y,cov,type='exact')
    #vssgpr.cov.update_periods(np.array([np.inf,np.inf]))
    print vssgpr.ELBO(vssgpr.params,vssgpr.x_train,vssgpr.y_train)
    #print vssgpr.ELBO_by_batch(vssgpr.params,vssgpr.x_train,vssgpr.y_train)