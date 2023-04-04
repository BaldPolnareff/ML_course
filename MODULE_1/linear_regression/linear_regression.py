import numpy as np
import copy, math
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 
import seaborn as sns 
import ipywidgets 

#class regression_plot():
#    def __init__(self, data, bias, weights_range=[0, 400], N=1000) -> None:
#        self.D = data
#        self.W = torch.linspace(weights_range[0], weights_range[-1], N)
#        self.bias = bias
#        self.B = self.bias * torch.ones(N)
#        #print(f'self.bias={self.bias}')
#    def f_wb(self, x, w, b):
#        return w * x + b 
#
#    def MSE(self, x, y, w, b, fun=None):
#        self.m = x.shape[0]
#        if fun==None: 
#            fun = self.f_wb
#        self.y_hat = fun(x, w, b)
#        cost = 1/(2*self.m) * torch.sum((self.y_hat - y)**2)
#        return cost 
#
#    def interact_plot_funcs(self, fun1=None, fun2=None):
#        self.X = self.D[0]
#        self.Y = self.D[-1]
#        if fun1 == None: 
#            fun1 = self.f_wb
#        if fun2 == None: 
#            fun2 = self.MSE
#        self.Y_hat = fun1(x=self.X, w=self.W, b=self.bias)
#        
#
#        def plot_funcs(self, x=self.X, y=self.Y, y_hat=self.Y_hat, w=100, size=(12, 5), title0='Linear Regression', 
#                       xlabel0='Size [1000 ft²]', ylabel0='Prices [1000 USD]', leg0=['data points', 'linear regression'], 
#                       title1='Cost Function', xlabel1='w', ylabel1='J(w)', leg1=['cost', 'current cost']):
#            self.fig, self.axs = plt.subplots(nrows=1, ncols=2, figsize=size)
#            self.axs[0].plot(x.numpy(), x.numpy(), 'o')
#            self.axs[0].plot(x.numpy(), y_hat.numpy(), '-')
#            self.axs[0].set_title(title0)
#            self.axs[0].set_xlabel(xlabel0)
#            self.axs[0].set_ylabel(ylabel0)
#            self.axs[0].legend(leg0)
#
#            self.J = torch.tensor([fun2(fun1, w, self.B[i], self.X, self.Y) for i, w in enumerate(self.W)])
#
#            self.axs[1].plot(self.W.numpy(), self.J.numpy(), '-')
#            self.axs[1].scatter(w, fun2(fun1, w, self.B[0], self.X, self.Y), c='r', marker='o')
#            self.axs[1].set_title(title1)
#            self.axs[1].set_xlabel(xlabel1)
#            self.axs[1].set_ylabel(ylabel1)
#            self.axs[1].legend(leg1)
#            plt.show()
#        
#        interact = ipywidgets.interact(plot_funcs, X=ipywidgets.fixed(self.X), Y=ipywidgets.fixed(self.Y), w=(self.W[0], self.W[-1], 0.1))
#        return interact

class linear_regression():
    def __init__(self, X, y):
       self.X = X 
       self.y = y
        
    def f_wb(self, w, b, X):
        return np.dot(w, X) + b 
    
    def MSE_cost(self, X, y, w, b): 
        m = X.shape[0]
        cost = 0
        for i in range(m):
            y_hat = self.f_wb(w, b, X[i])
            cost += (y_hat - y[i])**2
        cost *= 1/(2*m)
        return cost
    
    def compute_gradient(X, y, w, b): 
        """
        Computes the gradient for linear regression 
        Args:
          X (ndarray (m,n)): Data, m examples with n features
          y (ndarray (m,)) : target values
          w (ndarray (n,)) : model parameters  
          b (scalar)       : model parameter

        Returns:
          dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
          dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
        """
        m,n = X.shape           #(number of examples, number of features)
        dj_dw = np.zeros((n,))
        dj_db = 0.

        for i in range(m):                             
            err = (np.dot(X[i], w) + b) - y[i]   
            for j in range(n):                         
                dj_dw[j] = dj_dw[j] + err * X[i, j]    
            dj_db = dj_db + err                        
        dj_dw = dj_dw / m                                
        dj_db = dj_db / m                                

        return dj_db, dj_dw

    def gradient_descent(self, w_in, b_in, X, y, costfun=None, gradientfun=None, learning_rate=5.0e-7, N_iter=1000):
        J_history = [] # storing cost values per iteration to plot later
        alpha = learning_rate
        N = N_iter
        w = copy.deepcopy(w_in) # not changing input directly
        b = b_in 

        for i in range(N):
            if costfun == None:
                costfun = self.MSE_cost
            if gradientfun == None:
                gradientfun = self.compute_gradient 
            dj_db, dj_dw = gradientfun(X, y, w, b) # calculate the gradients 

            w = w - alpha * dj_dw # updating the params simultaneously
            b = b - alpha * dj_db # updating the params simultaneously

            if i < 10e5: # prevents resources exhaustion
                J_history.append(costfun(X, y, w, b))

            if i % math.ceil(N/10) == 0: # printing history every 10 iterations or as many iterations if < 10
                print(f'Iteration {i:4d}: Cost {J_history[-1]:8.2f}  ')

        return w, b, J_history

            

        

