
# coding: utf-8

# In[12]:


import preprocess_new
from sklearn.feature_extraction.text import TfidfVectorizer


# In[13]:

#Defining a Custom Vectorizer for Sklearn framework using TfidfVectorizer as Underlying engine and our preProcess method
class CustomVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(CustomVectorizer, self).build_tokenizer()
        return lambda doc: list(preprocess_new.preProcess(doc))


# In[14]:


def getBoWTfidfVectors(text_ary):
    preProcessedTrDF= preprocess_new.prepareTrainTestSet('train.csv','test.csv','bow',seperateLabelInfo=1,tokenize=0)
    boWTfidfVectorizer = CustomVectorizer()
    vectors = boWTfidfVectorizer.fit_transform(preProcessedTrDF['comment_text'])
    return vectors