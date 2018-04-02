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

import pickle
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix

#run this only the first time to save the vectors
'''
preProcessedTrainDF = pre.prepareTrainTestSet('train.csv','test.csv','word2vec',seperateLabelInfo=1)
sent2vec = []
sent2vec = wtv.WordsToVecFunction(preProcessedTrainDF)
'''


####################################################################
#load the vectors and create the dataset
####################################################################

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

sb.heatmap(data.corr()) 

#Except for the main diagonal, for which the color suggests that the correlation 
#of the variables with themselves is 1, the rest of the matrix is characterised 
#by features that present very low correlations between each other.
   
   

########################################################################################################
########################################################################################################
#TARGET: TOXIC (_1)
########################################################################################################
########################################################################################################

#Logistic regression in sklearn doesn't support multilabel classification  

####################################################################
#Dataset preparation
#################################################################### 

X = data.values
y_1 = []
for el in sent2vec:
    y_1.append(el[1][0]) #el[1] is the list of the 6 labels; el[1][0] is the first label corresponding to the toxic class
print(y_1[:20])
   
   
####################################################################
#Analysis of target
#################################################################### 

print(plt.hist(y_1))
print("\nPercentage of toxic comments: "+str(round(sum(y_1)/len(y_1),1)*100))
#The target is strongly unbalanced.


X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y_1, test_size = .3, random_state=25)


####################################################################
#ModelA
#################################################################### 
​
LogRegA = LogisticRegression() 
#class_weight parameter set as default: all classes have weight 1
#multi_class parameter set as default ('ovr'): binary problem set for each label 
LogRegA.fit(X_train_1, y_train_1)
​
y_predA = LogRegA.predict(X_test_1)
​

confusion_matrixA = confusion_matrix(y_test_1, y_predA)
print(confusion_matrixA)

print("Number of samples in the test set: "+str(len(y_test_1)))
print("Number of toxic comments (target = 1) in the test set: "+str(sum(y_test_1))+" ("+str(round(sum(y_test_1)/len(y_test_1)*100,2))+"% of the total)")

#All toxic comments are predicted as non-toxic. The model is not working as expected but the accuracy is quite high.

print(skm.classification_report(y_test_1, y_predA))

accA = skm.accuracy_score(y_test_1, y_predA, normalize=True, sample_weight=None)
print("Accuracy: "+str(round(accA,2)*100)+'%')



####################################################################
#Balancing - ModelB
#################################################################### 

LogRegB = LogisticRegression(class_weight = 'balanced') #adjust weights inversely proportional to class frequencies in the input data
LogRegB.fit(X_train_1, y_train_1)
​
y_predB = LogRegB.predict(X_test_1)
​

confusion_matrixB = confusion_matrix(y_test_1, y_predB)
print(confusion_matrixB)

print(skm.classification_report(y_test_1, y_predB))

acc2 = skm.accuracy_score(y_test_1, y_predB, normalize=True, sample_weight=None)
print("Accuracy: "+str(round(acc2,2)*100)+'%')



#In this case the accuracy is lower compared to the first model, but this works better if we consider
#other evaluation metrics, for example recall for class 1 (or true positive rate): 
#TPR for the first model is 0 (no toxic comment correctly classified), for model2 is
#0.55. Since our aim is to identify toxic comments, the second model is absolutely better than 
#the first one. 




########################################################################################################
########################################################################################################
#TARGET: SEVERE TOXIC (_2) / ModelC
########################################################################################################
########################################################################################################

y_2 = []
for el in sent2vec:
    y_2.append(el[1][1]) #el[1] is the list of the 6 labels; el[1][1] is the second label corresponding to the severe-toxic class
print(y_2[:20])

#analysis of the target 
print(plt.hist(y_2))
print("\nPercentage of severe toxic comments: "+str(round(sum(y_2)/len(y_2),3)*100))
#The target is strongly unbalanced.

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y_2, test_size = .3, random_state=25)

LogRegC = LogisticRegression(class_weight = 'balanced') 
LogRegC.fit(X_train_2, y_train_2)
​
y_predC = LogRegC.predict(X_test_2)
​

confusion_matrixC = confusion_matrix(y_test_2, y_predC)
print(confusion_matrixC)

print("Number of samples in the test set: "+str(len(y_test_2)))
print("Number of severe-toxic comments (target = 1) in the test set: "+str(sum(y_test_2))+" ("+str(round(sum(y_test_2)/len(y_test_2)*100,2))+"% of the total)")

print(skm.classification_report(y_test_2, y_predC))

accC = skm.accuracy_score(y_test_2, y_predC, normalize=True, sample_weight=None)
print("Accuracy: "+str(round(accC,2)*100)+'%')



########################################################################################################
########################################################################################################
#TARGET: OBSCENE (_3) / ModelD
########################################################################################################
########################################################################################################

y_3 = []
for el in sent2vec:
    y_3.append(el[1][2]) #el[1] is the list of the 6 labels; el[1][2] is the third label corresponding to the obscene class
print(y_3[:20])

#analysis of the target 
print(plt.hist(y_3))
print("\nPercentage of obscene comments: "+str(round(sum(y_3)/len(y_3),3)*100))
#The target is strongly unbalanced.

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y_3, test_size = .3, random_state=25)

LogRegD = LogisticRegression(class_weight = 'balanced') 
LogRegD.fit(X_train_3, y_train_3)
​
y_predD = LogRegD.predict(X_test_3)
​

confusion_matrixD = confusion_matrix(y_test_3, y_predD)
print(confusion_matrixD)

print("Number of samples in the test set: "+str(len(y_test_3)))
print("Number of obscene comments (target = 1) in the test set: "+str(sum(y_test_3))+" ("+str(round(sum(y_test_3)/len(y_test_3)*100,2))+"% of the total)")

print(skm.classification_report(y_test_3, y_predD))

accD = skm.accuracy_score(y_test_3, y_predD, normalize=True, sample_weight=None)
print("Accuracy: "+str(round(accD,2)*100)+'%')


########################################################################################################
########################################################################################################
#TARGET: THREAT (_4) / ModelE
########################################################################################################
########################################################################################################

y_4 = []
for el in sent2vec:
    y_4.append(el[1][3]) #el[1] is the list of the 6 labels; el[1][3] is the fourth label corresponding to the threat class
print(y_4[:20])

#analysis of the target 
print(plt.hist(y_4))
print("\nPercentage of threat comments: "+str(round(sum(y_4)/len(y_4),3)*100))
#The target is strongly unbalanced.

X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X, y_4, test_size = .3, random_state=25)

LogRegE = LogisticRegression(class_weight = 'balanced') 
LogRegE.fit(X_train_4, y_train_4)
​
y_predE = LogRegE.predict(X_test_4)
​

confusion_matrixE = confusion_matrix(y_test_4, y_predE)
print(confusion_matrixE)

print("Number of samples in the test set: "+str(len(y_test_4)))
print("Number of threat comments (target = 1) in the test set: "+str(sum(y_test_4))+" ("+str(round(sum(y_test_4)/len(y_test_4)*100,2))+"% of the total)")

print(skm.classification_report(y_test_4, y_predE))

accE = skm.accuracy_score(y_test_4, y_predE, normalize=True, sample_weight=None)
print("Accuracy: "+str(round(accE,2)*100)+'%')



########################################################################################################
########################################################################################################
#TARGET: INSULT (_5) / ModelF
########################################################################################################
########################################################################################################

y_5 = []
for el in sent2vec:
    y_5.append(el[1][4]) #el[1] is the list of the 6 labels; el[1][4] is the fifth label corresponding to the insult class
print(y_5[:20])

#analysis of the target 
print(plt.hist(y_5))
print("\nPercentage of insult comments: "+str(round(sum(y_5)/len(y_5),3)*100))
#The target is strongly unbalanced.

X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X, y_5, test_size = .3, random_state=25)

LogRegF = LogisticRegression(class_weight = 'balanced') 
LogRegF.fit(X_train_5, y_train_5)
​
y_predF = LogRegF.predict(X_test_5)
​

confusion_matrixF = confusion_matrix(y_test_5, y_predF)
print(confusion_matrixF)

print("Number of samples in the test set: "+str(len(y_test_5)))
print("Number of insult comments (target = 1) in the test set: "+str(sum(y_test_5))+" ("+str(round(sum(y_test_5)/len(y_test_5)*100,2))+"% of the total)")

print(skm.classification_report(y_test_5, y_predF))

accF = skm.accuracy_score(y_test_5, y_predF, normalize=True, sample_weight=None)
print("Accuracy: "+str(round(accF,2)*100)+'%')







########################################################################################################
########################################################################################################
#TARGET: IDENTITY_HATE (_6) / ModelG
########################################################################################################
########################################################################################################

#an observation has 'nan' as the value for the identity-hate class. We remove it from the model
rows = []
y_6 = []
for i in range(len(sent2vec)):
    vect_components = []
    if sent2vec[i][1][5] in (0.0,1.0):
        for j in sent2vec[i][0]:        
            vect_components.append(j)
        rows.append(vect_components)
        y_6.append(sent2vec[i][1][5].astype(int)) #el[1] is the list of the 6 labels; el[1][5] is the sixth label corresponding to the identity-hate class
data = pd.DataFrame(rows)
print(data.shape) 
print(y_6[:20])



#analysis of the target 
print(plt.hist(y_6))
print("\nPercentage of insult comments: "+str(round((sum(y_6)/len(y_6))*100,2)))
#The target is strongly unbalanced.


X_new = data.values
X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(X_new, y_6, test_size = .3, random_state=25)

LogRegG = LogisticRegression(class_weight = 'balanced') 
LogRegG.fit(X_train_6, y_train_6)
​
y_predG = LogRegG.predict(X_test_6)
​

confusion_matrixG = confusion_matrix(y_test_6, y_predG)
print(confusion_matrixG)

print("Number of samples in the test set: "+str(len(y_test_6)))
print("Number of insult comments (target = 1) in the test set: "+str(sum(y_test_6))+" ("+str(round(sum(y_test_6)/len(y_test_6)*100,2))+"% of the total)")

print(skm.classification_report(y_test_6, y_predG))

accG = skm.accuracy_score(y_test_6, y_predG, normalize=True, sample_weight=None)
print("Accuracy: "+str(round(accG,2)*100)+'%')


#POSSIBLE IMPROVEMENTS
#We can test different values of threshold (probability of belonging to the target class
#compared to the other class) to select when an observation should be labelled as 1 or 0
#(default threshold is 0.5). 















   
   
