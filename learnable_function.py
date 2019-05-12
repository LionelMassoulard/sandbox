# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:30:49 2019

@author: Lionel Massoulard
"""

class Phase:
    train = "train"
    pred  = "pred"

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

class AutoModel(object):
    
    def __init__(self, name, choices, func):
        self.name = name
        self.choices = choices
        self.func = func
        
        self.trace = []
        self.forest = None
        
    def __call__(self, **kwargs):
        if self.func.phase == Phase.train:
            random_choice = self.choices[np.random.choice(len(self.choices))]
            
            self.trace.append( {"random_choice":random_choice,
                                "kwargs":kwargs,
                                "run_id":self.func.run_id})
    
            
            return random_choice
        else:
            dfX = pd.DataFrame([kwargs]*len(self.choices))
            dfX["random_choice"] = self.choices
            
            yhat = self.forest.predict(dfX)
            
            i = yhat.argmax()
            
            return self.choices[i]
            
            
    def fit(self):
        
        
        df_reward = pd.DataFrame(self.func.rewards)

        df_trace = pd.DataFrame([t["kwargs"] for t in self.trace])
        df_trace["random_choice"] = [t["random_choice"] for t in self.trace]
        df_trace["run_id"] = [t["run_id"] for t in self.trace]
        
        df_merged = pd.merge(left=df_trace,right=df_reward,on="run_id")
        
        
        
        dfX = df_merged.loc[:,[c for c in df_merged.columns if c not in ("run_id","reward")]]
        y   = df_merged["reward"]
        
        self.forest = RandomForestRegressor(n_estimators=100)
        self.forest.fit(dfX,y)

#forest.predict()
#
#test = pd.DataFrame({"X_size":[0,0,1,1,2,2,3,3],"choice":[0,1,0,1,0,1,0,1]})
#test["is_correct"] = forest.predict(test)
#
#        

class Learnable(object):
    
    def __init__(self):
        self.auto_models = {}
        self.phase = None
        self.run_id = None
        
        self.rewards = []
    
    def AutoTest(self,name, choices):
        if name not in self.auto_models:
            self.auto_models[name] = AutoModel(name=name, choices=choices, func=self)

        return self.auto_models[name]



        
#    def f(self,X):
#
#        c = self.AutoTest(name = "condition1", choices=[0,1])(X_size = len(X))
#        c = self.AutoTest(name = "condition1", choices=[0,1])(X_size = len(X))
#        
#        if c == 0:
#            return 1
#        else:
#            return 0
        
    def f(self, X):
        nb = 0
        while True:
            for i in range(len(X)-1):
                if self.AutoTest(name="c1",choices=[True,False])(X1=X[i-1],X2=X[i]):
                    X[i],X[i-1] = X[i-1], X[i]
                    nb += 1
                    
            if self.AutoTest(name="final",choices=[True,False])(nb=nb):
                break
            
            if nb >= 1000:
                break
            
        return X

    def save_reward(self,reward):
        self.rewards.append({"run_id":self.run_id,"reward":reward})

def get_reward(Ypred, Y):
    return 1*(Ypred == Y)
    
def get_truth(X):
    return 1*(len(X) == 2)

# In[]
Xs = [list(np.random.randint(0,5,np.random.randint(10))) for _ in range(100)]
Ys = [sorted(X) for X in Xs]

def get_reward(Ypred, Y):
    r = 0
    for i in range(len(Ypred)-1):
        r += Ypred[i+1] >= Ypred[i]
    return r/len(Ypred)
        


#Xs = [[1,2],[2,4],[],[3,4,5,6],[1]]
#Ys = [get_truth(X) for X in Xs]

FF = Learnable()
FF.phase = Phase.train

run_id = 0
for X,Y in zip(Xs,Ys):
    for b in range(100):
        FF.run_id = run_id
        Ypred = FF.f(X)
        reward = get_reward(Ypred,Y)
        FF.save_reward(reward)
        
        run_id += 1
        
        
for name, aut_model in FF.auto_models.items():
    aut_model.fit()

FF.phase = Phase.pred


