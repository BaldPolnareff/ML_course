import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 
import seaborn as sns 
import ipywidgets 

class regression_plot():
    def __init__(self, data, bias, weights_range=[0, 400], N=1000) -> None:
        self.D = data
        self.W = torch.linspace(weights_range[0], weights_range[-1], N)
        self.bias = bias
        self.B = self.bias * torch.ones(N)
        #print(f'self.bias={self.bias}')
    def f_wb(self, x, w, b):
        return w * x + b 

    def MSE(self, x, y, w, b, fun=None):
        self.m = x.shape[0]
        if fun==None: 
            fun = self.f_wb
        self.y_hat = fun(x, w, b)
        cost = 1/(2*self.m) * torch.sum((self.y_hat - y)**2)
        return cost 

    def interact_plot_funcs(self, fun1=None, fun2=None):
        self.X = self.D[0]
        self.Y = self.D[-1]
        if fun1 == None: 
            fun1 = self.f_wb
        if fun2 == None: 
            fun2 = self.MSE
        self.Y_hat = fun1(x=self.X, w=self.W, b=self.bias)
        

        def plot_funcs(self, x=self.X, y=self.Y, y_hat=self.Y_hat, w=100, size=(12, 5), title0='Linear Regression', 
                       xlabel0='Size [1000 ft²]', ylabel0='Prices [1000 USD]', leg0=['data points', 'linear regression'], 
                       title1='Cost Function', xlabel1='w', ylabel1='J(w)', leg1=['cost', 'current cost']):
            self.fig, self.axs = plt.subplots(nrows=1, ncols=2, figsize=size)
            self.axs[0].plot(x.numpy(), x.numpy(), 'o')
            self.axs[0].plot(x.numpy(), y_hat.numpy(), '-')
            self.axs[0].set_title(title0)
            self.axs[0].set_xlabel(xlabel0)
            self.axs[0].set_ylabel(ylabel0)
            self.axs[0].legend(leg0)

            self.J = torch.tensor([fun2(fun1, w, self.B[i], self.X, self.Y) for i, w in enumerate(self.W)])

            self.axs[1].plot(self.W.numpy(), self.J.numpy(), '-')
            self.axs[1].scatter(w, fun2(fun1, w, self.B[0], self.X, self.Y), c='r', marker='o')
            self.axs[1].set_title(title1)
            self.axs[1].set_xlabel(xlabel1)
            self.axs[1].set_ylabel(ylabel1)
            self.axs[1].legend(leg1)
            plt.show()
        
        interact = ipywidgets.interact(plot_funcs, X=ipywidgets.fixed(self.X), Y=ipywidgets.fixed(self.Y), w=(self.W[0], self.W[-1], 0.1))
        return interact
