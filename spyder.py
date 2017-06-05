#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 19:58:14 2017

@author: siddharth
"""

#importing essential python libraries
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
#readind and loading file
loc_train='/home/siddharth/Documents/Bike Renting/train.csv'
loc_test='/home/siddharth/Documents/Bike Renting/test.csv'
train=pd.read_csv(loc_train)
test=pd.read_csv(loc_test)
print (train.info())
target=train['count']
casual=train.casual
registered=train.registered
datetime=train.datetime


train.drop('count',axis=1, inplace=True)
train.drop('casual',axis=1,inplace=True)
train.drop('registered',axis=1,inplace=True)


combined=train.append(test)
combined.reset_index(inplace=True)
combined.drop('index',axis=1,inplace=True)

#All columns are numeric. No missing data present.
#12 columns in total
#Season,holiday,workingday,weather,temp,atemp,humidity,windspeed,casual,registered requires no formatting.


print (train.shape)
print (test.shape)
combined['datetime']=combined['datetime'].map(lambda dtime: dtime.split(' ')[1].split(':')[0])
combined['datetime']=combined['datetime'].astype(int)
print (combined.head(2))
combined.head()


train_new=combined[:10886]
test_new=combined[10886:]

#print (combined.info())
#print (train_new.info())
#print (test_new.info())
print (target.shape)
target
clf=RandomForestClassifier(n_estimators=70)
clf.fit(train_new,target)
pred=clf.predict(test_new)
output=pd.DataFrame()
output['datetime']=test['datetime']
output['count']=pred
output.to_csv('/home/siddharth/Documents/Bike Renting/output1.csv',index=False)