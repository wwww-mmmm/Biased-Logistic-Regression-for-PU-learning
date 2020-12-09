# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:17:13 2020

@author: 27264
"""

import numpy as np
from collections import Counter





class BinaryBiasLR():
    def __init__(self,learning_rate=0.01,
                 num_iteration=2000,
                 lambda_l2=0.001,
                 P_pre=0.33,
                 random_state = 2020,
                 verbose = None):
        
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.lambda_l2 = lambda_l2
        self.P_pre = P_pre
        self.verbose = verbose
        self.coef_ = None
        self.random_state = random_state
    
    '''
    The cost function and gradient is absolutely based on eassy
    
    ###Parameter:
    lambda_l2: only this method
    P_pre: accelerate convergence with the value increased, 
            means the proportion of the i-2 iteration's gradient is used
            to update the gradient of current(i) iteration's gradient
    '''
    
    
    
    
    def _get_w(self,x,y,weights):
        ## positive
        pos_idx = np.squeeze(np.argwhere(y==1))
        pos_x = x[pos_idx,:]
        exp_y_w_x = np.exp(-1*weights.dot(pos_x.transpose()))
        pos_grad = (-1)*exp_y_w_x/(1+exp_y_w_x)
        pos_grad = pos_grad.dot(pos_x)
        
        ## negative
        neg_idx = np.squeeze(np.argwhere(y==-1))
        neg_x = x[neg_idx,:]
        exp_y_w_x = np.exp(1*weights.dot(neg_x.transpose()))
        neg_grad = (1)*exp_y_w_x/(1+exp_y_w_x)
        neg_grad = neg_grad.dot(neg_x)      
        
        
        w_record = pos_grad*len(neg_idx)/len(pos_idx)+neg_grad
        return w_record
        
  
    
    
    
        
    def get_loss(self,x,y,weights):
        pos_idx = np.squeeze(np.argwhere(y==1))
#        print(weights.dot(x[pos_idx,:].transpose()))
        pos_loss = np.log(np.exp(-1*np.squeeze(y[pos_idx])*weights.dot(x[pos_idx,:].transpose()))+1)
        
        neg_idx = list( set(range(x.shape[0])) - set(pos_idx) )
        neg_loss = np.log(np.exp(-1*np.squeeze(y[neg_idx])*weights.dot(x[neg_idx,:].transpose()))+1)
        pos_loss = len(neg_idx)*sum(pos_loss)/len(pos_idx)
 
        neg_loss = sum(neg_loss)
        
        w_loss = self.lambda_l2*sum(weights**2)
        
        return pos_loss,neg_loss,w_loss
        
        
        
  
    
    def fit(self,train_x,train_y):
        np.random.seed(self.random_state)
        if not isinstance(train_x,np.ndarray):
            train_x = np.array(train_x)
        if not isinstance(train_y,np.ndarray):
            train_y = np.array(train_y)
        train_y = np.squeeze(train_y)
        y_count = Counter(train_y)
        y_keys = list(y_count.keys())
        assert y_keys[0]*y_keys[1]==-1 and sum(y_keys)==0 and len(y_keys)==2, '{positve:1,negative:-1}'
        
        ### 加入偏置
        new_x = np.array([1]*train_x.shape[0]).reshape(-1,1)
        new_x = np.hstack([new_x,train_x])
        pre_grad = np.array([0]*new_x.shape[1])
        
        
        ### 初始化权重
        if self.random_state:
            self.weights = np.random.normal(size = (new_x.shape[1],))
        else:
            self.weights = np.array([1]*new_x.shape[1])
        print(self.weights)
        itera = 0
        while itera<self.num_iteration:
            grad = self._get_w(new_x,train_y,self.weights)
            total_grad = grad+2*self.lambda_l2*self.weights
            
            self.weights = (1-self.lambda_l2)*self.weights - \
            self.learning_rate*(total_grad+self.P_pre*pre_grad)
            
            pre_grad = total_grad
            
            
            pos_loss,neg_loss,w_loss = self.get_loss(new_x,train_y,self.weights)
            total_loss = pos_loss+neg_loss+w_loss
            if self.verbose>0:
                if itera%self.verbose == 0:
                    print('iteration:%s|pos_loss:%s|neg_loss:%s|total_loss:%s'%(itera,
                                                                          round(pos_loss,4),
                                                                          round(neg_loss,4),
                                                                          round(total_loss,4)))
            itera+=1
        self.coef_ = self.weights[1:].tolist()

            
            
            
        
        
    def predict(self,test_x):
        new_test_x = np.array([1]*test_x.shape[0]).reshape(-1,1)
        new_test_x = np.hstack([new_test_x,test_x])
        self.prob = 1/(1+np.exp(-1*self.weights.dot(new_test_x.transpose())))
        self.res = np.where(self.prob>0.5, 1,-1)
        
        self.prob = np.hstack([1-self.prob.reshape(-1,1),self.prob.reshape(-1,1)])
        return self.res
    
    
    
    def predict_proba(self,test_x):
        self.predict(test_x)
        return self.prob







## example 

x = np.array([[3,2,1],
              [3,2,1],
              [3,2,1],
              [3,2,1],
              [1,3,3],
              [1,3,2],
              [1,3,2],
              [1,3,2]])
y = np.array([1,1,1,1,-1,-1,-1,-1])

lr = BinaryBiasLR(learning_rate=0.01,num_iteration=500,lambda_l2 = 0.001,
                  P_pre = 0.1,random_state = 2020,verbose = 1)
lr.fit(x,y)
weight = lr.coef_
print(weight)

y_prob = lr.predict_proba(x)
print(y_prob)

y_pred = lr.predict(x)
print(y_pred)

















