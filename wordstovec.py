import preprocess_new as pre
import gensim
import numpy
import csv
from nltk.corpus import brown, movie_reviews, treebank
from gensim.models import KeyedVectors
filename = '/Users/boutrosazar/Downloads/GoogleNews-vectors-negative300.bin'

#get the preprocessed words and their labels
preProcessedTrainDF = pre.prepareTrainTestSet('train.csv','test.csv','bow',seperateLabelInfo=1)#,sampleNum=10) #159548
numElements = len(preProcessedTrainDF)
myList = []
myLabels = []


for x in range(numElements):
    myList.append(preProcessedTrainDF.iloc[x][1])
    myLabels.append([preProcessedTrainDF.iloc[x][i] for i in range(2,8)])

#Train the model word2vec
model = gensim.models.Word2Vec(iter=1,min_count=0)
model.build_vocab(myList)

#Avg the words to have one vector per sentence
Sent2Vec = []
myListLen = len(myList)
mySum = numpy.zeros(100)

for j in range(myListLen):
    mySum = numpy.zeros(100)
    for i in myList[j]:
        mySum += model[i]
    myAvg = mySum/len(myList[j])
    Sent2Vec.append([myAvg, myLabels[j]])

#csv File contains the sentence as a vector and the labels
with open('trainVec.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(Sent2Vec)