# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
os.chdir('/Users/elenapedrini/Desktop/GitHub/QMULHub/')
import wordstovec as wtv
import preprocess_new as pre
import pandas as pd


#run this only the first time to save the vectors
'''
preProcessedTrainDF = pre.prepareTrainTestSet('train.csv','test.csv','word2vec',seperateLabelInfo=1)
sent2vec = []
sent2vec = wtv.WordsToVecFunction(preProcessedTrainDF)
'''


####################################################################
#load the vectors and create the dataset
####################################################################

import pickle
#with open("/Users/elenapedrini/Desktop/GitHub/sent2vec.txt", "wb") as fp:   #Pickling
 #  pickle.dump(sent2vec, fp)

with open("/Users/elenapedrini/Desktop/GitHub/sent2vec.txt", "rb") as fp:   # Unpickling
   sent2vec = pickle.load(fp)
   
   
   
rows = []
for i in range(len(sent2vec)):
    vect_components = []
    for j in sent2vec[i][0]:
        vect_components.append(j)
    rows.append(vect_components)
data = pd.DataFrame(rows)
print(data.shape) 

data.head()

   
####################################################################
#correlation
####################################################################
import seaborn as sb
sb.heatmap(data.corr()) 

#Except for the main diagonal, for which the color suggests that the correlation 
#of the variables with themselves is 1, the rest of the matrix is characterised 
#by features that present very low correlations between each other.
   
   

########################################################################################################
########################################################################################################
#TARGET: TOXIC
########################################################################################################
########################################################################################################
  
####################################################################
#Dataset preparation
#################################################################### 

X = data.values
y = []
for el in sent2vec:
    y.append(el[1][0]) #el[1] is the list of the 6 labels; el[1][0] is the first label corresponding to the toxic class
print(y[:20])
   
   
####################################################################
#Analysis of target
#################################################################### 

import matplotlib.pyplot as plt
print(plt.hist(y))
print("\nPercentage of toxic comments: "+str(round(sum(y)/len(y),1)*100))
   
   
#The target is strongly unbalanced.

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)



####################################################################
#Model
#################################################################### 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
​
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
​
y_pred = LogReg.predict(X_test)
​
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print("Number of samples in the test set: "+str(len(y_test)))
print("Number of toxic comments (target = 1) in the test set: "+str(sum(y_test))+" ("+str(round(sum(y_test)/len(y_test)*100,2))+"% of the total)")


#All toxic comments are predicted as non-toxic. The model is not working as expected but the accuracy is quite high.

import sklearn.metrics as skm

print(skm.classification_report(y_test, y_pred))
print("Accuracy: "+str(round(skm.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None),2)))












   
   