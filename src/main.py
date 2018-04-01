'''
Created on 2018年3月27日

@author: WZD
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc  
import matplotlib.pyplot as plt  
import random  

class LogisticRegression:
    def __init__(self,epoch=10000,solver='GA',learning_rate=0.001,threshold = 1e-4):    
        '''
        batch_size:the size of each batch
        epoch:max epoch of training
        solver: GA(gradient ascent) or NM(newton method)  '''        
        self.epoch = int(epoch)
        self.solver = solver                  
        self.theta = None  
        self.learning_rate = learning_rate   
        self.threshold = threshold
        self.cost = pd.Series(np.arange(self.epoch, dtype = float)) # cost function
        
    def fit(self,X,y):        
        X = np.insert(arr=X, obj=0, values=1, axis=1)
        n = X.shape[1] #the number of features        
        m = X.shape[0] #the number of samples      
        self.theta = np.zeros((n,1)) #initialize parameter
        
        if self.solver == 'GA':
            for i in range(self.epoch):
                y_pro = 1./(1.+np.exp(-np.dot(X,self.theta))) #probability                
                self.cost[i] = -(1/m)*np.sum(y*np.log(y_pro.flatten())+(1-y)*np.log(1-y_pro.flatten())) #computer value of cost function
                gradient = np.dot(X.T,y_pro-y[:,np.newaxis])                                                
                self.theta -= self.learning_rate*gradient
                if np.linalg.norm(self.theta)<self.threshold:
                    break
                
        elif self.solver == 'NM':
            for i in range(self.epoch):               
                y_pro = (1./(1.+np.exp(-np.dot(X,self.theta)))) # probability      
                self.cost[i] = -(1./m)*np.sum(y*np.log(y_pro.flatten())+(1-y)*np.log(1-y_pro.flatten())) # computer value of cost function
                gradient = np.dot(X.T,y[:,np.newaxis]-y_pro)                 
                A =  y_pro*(y_pro-1)* np.eye(len(X))  
                H = np.mat(X.T)* A * np.mat(X) # Hessian matrix, H = X`AX                  
                self.theta -= np.linalg.pinv(H)*gradient
                if np.linalg.norm(self.theta)<self.threshold:
                    break
                
                
        print(self.cost.values)
        plt.plot(self.cost.values,color='r')
        plt.xlabel(('epoch'))
        plt.ylabel('cost')        
        plt.show()
    
    def predict(self,X):        
        X = np.insert(arr=X, obj=0, values=1, axis=1)        
        y_pro = 1./(1.+np.exp(-np.dot(X,self.theta)))
        y_pre = np.zeros((X.shape[0],1),dtype = int)
        for i in range(y_pro.shape[0]):
            if y_pro[i] >0.5:
                y_pre[i]=1
            else:
                y_pre[i]=0                                
        return y_pro.flatten(),y_pre.flatten().astype(int)

if __name__ == '__main__':
     
    #===========================================================================
    # preprocessing
    #===========================================================================
      
    data = pd.read_csv('./Credit.csv',index_col=0)
    print(' print the fist-five lines of data')
    print(data.head(n=5))
    data = data.dropna(axis=0)
    y = data.values[:,0].astype(int) # one-dimensional array    
    X = data.drop(labels=['Label','Loanaccount'],axis=1)
    print('feature matrix')
    print(X.head(n=5))     
    X = pd.get_dummies(X).astype('float') #one-hot encoding for discrete features   
    X = (X-X.min())/(X.max()-X.min())
    X = X.values    
    #pca = PCA(n_components=10)
    #X = pca.fit_transform(X) 
    #split train set and test set
    trainX,testX,trainY,testY = train_test_split(X,y,test_size=0.33,random_state=42)
     
    #===========================================================================
    # train
    #===========================================================================
     
    lr = LogisticRegression(epoch=20, solver='NM', learning_rate=0.001,threshold=1e-4)      
    lr.fit(trainX,trainY)    
    #===========================================================================
    # test
    #===========================================================================
    y_pro,y_pre = lr.predict(testX) 
        
    #===========================================================================
    # evaluation
    #===========================================================================
    tn,fp,fn,tp = confusion_matrix(y_true=testY, y_pred=y_pre).ravel()
    print('准确率：',(tp+tn)/(tn+fp+fn+tp))
    print('查全率：',tp/(tp+fn))
    print('查准率：',tp/(tp+fp))    
    print('auc:',roc_auc_score(y_true=testY, y_score=y_pro))
    get_ks = lambda y_pred,y_true: ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic      
    print('ks:',get_ks(y_pre,testY))
    
    #plot ROC
    fpr,tpr,thresholds = roc_curve(y_true=testY, y_score=y_pro)
    roc_auc = auc(fpr,tpr)  
    plt.title('Receiver Operating Characteristic')  
    plt.plot(fpr,tpr,'b',label='AUC = %0.2f'% roc_auc)  
    plt.legend(loc='lower right')  
    plt.plot([0,1],[0,1],'r--')      
    plt.ylabel('True Positive Rate')  
    plt.xlabel('False Positive Rate')  
    plt.show()   
    
    #Somer’s D concordance statistics
    pr_0 = []
    pr_1 = []    
    for i in range (testY.shape[0]):
        if testY[i]==0:
            pr_0.append(y_pro[i])
        else:
            pr_1.append(y_pro[i])    
    score = 0.0
    for i in range(len(pr_1)):
        for j in range(len(pr_0)):
            if pr_1[i]>pr_0[j]:
                score += 1
            else:
                score +=-1
    print('Somer’s D concordance statistics:',score/(len(pr_1)*len(pr_0)))
    
            
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    

















