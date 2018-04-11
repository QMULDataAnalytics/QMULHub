
# coding: utf-8

# # READ DATA####

# Load the CSV file in a dataframe object.

# In[4]:


import pandas as pd
import re
import nltk
import string
#from nltk.corpus import stopwords


# In[5]:


trainDF,testDF = [],[]


# In[6]:


# Input : trainCsvPath, testCsvPath
# Output: (trainDF,testDF)
def loadData(trainCsvPath,testCsvPath):
    trainDF = pd.read_csv(trainCsvPath,header=0)
    testDF = pd.read_csv(testCsvPath,header=0)
    return(trainDF,testDF)


# In[7]:


# TEXT PREPROCESSING AND FEATURE VECTORIZATION

# Input: a string of one review
# Output: tokens of n-grams

def preProcess(text):
    # Should return a list of tokens
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append('') #to remove blank elements 
    lemmatizer = nltk.WordNetLemmatizer() #lemmatization 
    stemmer = nltk.PorterStemmer() #stemming
    
    #remove punctuation
    for el in text:
        if el in string.punctuation:
            text = text.replace(el, ' ')
        
    tokens = re.split(r"\s+",text) #tokenisation
    tokens = [t.lower() for t in tokens] #lower case
    tokens = [item for item in tokens if item not in stopwords] #remove stop words 
    tokens=[lemmatizer.lemmatize(t) for t in tokens] #lemmatizer 
    tokens = [stemmer.stem(t) for t in tokens] #stemmer

    unigrams = list(nltk.ngrams(tokens,1))
    #bigrams = list(nltk.ngrams(tokens,2))
    #trigrams =  list(nltk.ngrams(tokens,3))
    tokens = unigrams #+ bigrams #+ trigrams
    
    return tokens


# In[8]:


# Input : trainCsvPath, testCsvPath and the name of the Vectorization Method such as ("bow","glove","word2vec"), seperate Label info if 1, num of Samples to process
# Output: the train(with label) and test dataset in internal representation
from tqdm import tqdm
def prepareTrainTestSet(trainCsvPath,testCsvPath,vectorizationMethod,seperateLabelInfo=0,sampleNum=0,tokenize=1):
    #Load Data
    global trainDF,testDF
    trainDF,testDF = loadData(trainCsvPath,testCsvPath)
    
    if (not sampleNum == 0):
        trainDF=trainDF[0:sampleNum]
        testDF=testDF[0:sampleNum]
    
    #preProcess the train Dataset
    def iterateDF(df):
        numElements = len(df)
        for i in tqdm(range(0,numElements)):
            df.iloc[i,df.columns.get_loc('comment_text')]=preProcess(df.iloc[i,df.columns.get_loc('comment_text')])
        return df
    if(tokenize==1):
        preProcessedTrainDF=iterateDF(trainDF)
    else:
        preProcessedTrainDF=trainDF
    return preProcessedTrainDF


# In[10]:


#preProcessedTrainDF= prepareTrainTestSet('train.csv','test.csv','bow',seperateLabelInfo=1,sampleNum=100)


# In[11]:


#preProcessedTrainDF.iloc[1]#,train.columns.get_loc('comment_text')]
#train.iloc[1]#.loc['comment_text']
#preProcess(train.iloc[1,train.columns.get_loc('comment_text')])

