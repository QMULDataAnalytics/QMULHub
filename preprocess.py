
# coding: utf-8

# # READ DATA####

# Load the CSV file in a dataframe object.

# In[1]:


import pandas as pd
import re
import nltk
#from nltk.corpus import stopwords


# In[2]:


trainDF,testDF = [],[]


# In[3]:


# Input : trainCsvPath, testCsvPath
# Output: (trainDF,testDF)
def loadData(trainCsvPath,testCsvPath):
    trainDF = pd.read_csv(trainCsvPath,header=0)
    testDF = pd.read_csv(testCsvPath,header=0)
    return(trainDF,testDF)


# In[4]:


# TEXT PREPROCESSING AND FEATURE VECTORIZATION

# Input: a string of one review
# Output: tokens of n-grams
def preProcess(text):
    # Should return a list of tokens
    # word tokenisation
    text = re.sub(r"(\w)([.,;:!?'\"”\)])", r"\1 \2", text)
    text = re.sub(r"([.,;:!?'\"“\(])(\w)", r"\1 \2", text)
    text = re.sub(r"(\S)\1\1+",r"\1\1\1", text)
    tokens = re.split(r"\s+",text)
    # normalisation
    tokens = [t.lower() for t in tokens]
    #stop word removal
    stop = nltk.corpus.stopwords.words('english')
    tokens = [item for item in tokens if item not in stop]
    #stemmer
    porter = nltk.PorterStemmer()
    tokens = [porter.stem(t) for t in tokens]
    #lemmatizer
    wnl = nltk.WordNetLemmatizer()
    tokens=[wnl.lemmatize(t) for t in tokens]
    #bigrams
    unigrams = list(nltk.ngrams(tokens,1))
    #bigrams = list(nltk.ngrams(tokens,2))
    #trigrams =  list(nltk.ngrams(tokens,3))
    tokens = unigrams #+ bigrams + trigrams
    return tokens


# In[5]:


# Input : trainCsvPath, testCsvPath and the name of the Vectorization Method such as ("bow","glove","word2vec"), seperate Label info if 1, num of Samples to process
# Output: the train(with label) and test dataset in internal representation
def prepareTrainTestSet(trainCsvPath,testCsvPath,vectorizationMethod,seperateLabelInfo=0,sampleNum=0):
    #Load Data
    global trainDF,testDF
    trainDF,testDF = loadData(trainCsvPath,testCsvPath)
    
    if (not sampleNum == 0):
        trainDF=trainDF[0:sampleNum]
        testDF=testDF[0:sampleNum]
    
    #preProcess the train Dataset
    def iterateDF(df):
        numElements = len(df)
        for i in range(0,numElements):
            df.iloc[i,df.columns.get_loc('comment_text')]=preProcess(df.iloc[i,df.columns.get_loc('comment_text')])
        return df
    preProcessedTrainDF=iterateDF(trainDF)
    return preProcessedTrainDF


# In[6]:


preProcessedTrainDF= prepareTrainTestSet('train.csv','test.csv','bow',seperateLabelInfo=1,sampleNum=100)


# In[13]:


preProcessedTrainDF.iloc[1]#,train.columns.get_loc('comment_text')]
#train.columns.get_loc('comment_text')
#train.iloc[1].loc['comment_text']
#preProcess(train.iloc[1,train.columns.get_loc('comment_text')])

