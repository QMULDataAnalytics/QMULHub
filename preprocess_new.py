import pandas as pd
import re
import nltk
#from nltk.corpus import stopwords

trainDF,testDF = [],[]

# Input : trainCsvPath, testCsvPath
# Output: (trainDF,testDF)
def loadData(trainCsvPath,testCsvPath):
    trainDF = pd.read_csv(trainCsvPath,header=0)
    testDF = pd.read_csv(testCsvPath,header=0)
    return(trainDF,testDF)



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
    bigrams = list(nltk.ngrams(tokens,2))
    #trigrams =  list(nltk.ngrams(tokens,3))
    tokens = unigrams + bigrams #+ trigrams
    
    return tokens



# Input : trainCsvPath, testCsvPath and the name of the Vectorization Method such as ("bow","glove","word2vec"), seperate Label info if 1, num of Samples to process
# Output: the train(with label) and test dataset in internal representation
def prepareTrainTestSet(trainCsvPath,testCsvPath,vectorizationMethod,seperateLabelInfo=0,sampleNum=0):
    #Load Data
    global trainDF,testDF
    trainDF,testDF = loadData(trainCsvPath,testCsvPath)
    
    if (not sampleNum == 0):
        trainDF=trainDF[1:sampleNum]
        testDF=testDF[1:sampleNum]
    
    #preProcess the train Dataset
    def iterateDF(df):
        numElements = len(df)
        for i in range(0,numElements):
            df.iloc[i,df.columns.get_loc('comment_text')]=preProcess(df.iloc[i,df.columns.get_loc('comment_text')])
        return df
    preProcessedTrainDF=iterateDF(trainDF)
    return preProcessedTrainDF




preProcessedTrainDF= prepareTrainTestSet('train.csv','test.csv','bow',seperateLabelInfo=1,sampleNum=100)

preProcessedTrainDF.iloc[1]#,train.columns.get_loc('comment_text')]
#train.iloc[1]#.loc['comment_text']
#preProcess(train.iloc[1,train.columns.get_loc('comment_text')])
