#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle 

# In[3]:


df = pd.read_csv("nlp dataset/training_set_rel3.tsv", sep='\t', encoding='ISO-8859-1')


# In[4]:


df = df.dropna(axis=1)


# In[5]:


df = df.drop(columns=['rater1_domain1', 'rater2_domain1'])


# In[6]:


y = df['domain1_score']


# In[7]:


df.head()


# In[8]:


def word(sentence):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}')
    stop_word = set(stopwords.words("english"))
    words = tokenizer.tokenize(sentence)
    words=[i.lower() for i in words]
    words = [w for w in words if not w in stop_word]
    return words
def sentence(essay):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    s = tokenizer.tokenize(essay)
    fs = []
    for i in s:
        fs.append(word(i))
    return fs
def word_vectors(words,model,dim):
    fv = np.zeros((dim,),dtype='float32')
    i2w_set = set(model.wv.index2word)
    num_words = 0
    for word in words:
        if word in i2w_set:
            num_words+=1
            fv = np.add(fv,model[word])
    fv = np.divide(fv,num_words)
    return fv
def avg_word_vectors(essay,model,dim):
    efv = np.zeros((len(essay),dim), dtype='float32')
    c=0
    for i in essay:
        efv[c] = word_vectors(i,model,dim)
        c+=1
    return efv


# In[10]:


from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K

def get_model():
    """Define the model."""
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score
from gensim.models import Word2Vec

sent_list = []
results=[]
cv = KFold(5, shuffle=True)
count=1
pkl_filename="nlp_model.pkl"
for train,test in cv.split(df):
    x_train,x_test,y_train,y_test = df.iloc[train], df.iloc[test], y.iloc[train], y.iloc[test]
    train_set = x_train['essay']
    test_set = x_test['essay']
    for i in train_set:
        sent_list+=sentence(i)
    
    model = Word2Vec(sent_list, workers=4, size=300, min_count = 40, window = 10, sample = 1e-3)
    model.wv.save_word2vec_format('word2vecmodel.bin', binary=True)
    essay_train_words = []
    for i in train_set:
        essay_train_words.append(word(i))
    train_vec = avg_word_vectors(essay_train_words,model,300)
    essay_test_words = []
    for i in test_set:
        essay_test_words.append(word(i))
    test_vec = avg_word_vectors(essay_test_words,model,300)
    train_vec = np.array(train_vec)
    test_vec = np.array(test_vec)
    train_vec = np.reshape(train_vec, (train_vec.shape[0], 1, train_vec.shape[1]))
    test_vec = np.reshape(test_vec, (test_vec.shape[0], 1, test_vec.shape[1]))
    
    lstm_model = get_model()
    lstm_model.fit(train_vec, y_train, batch_size=64, epochs=50)
    y_pred = lstm_model.predict(test_vec)
    if count == 5:
         lstm_model.save('./final_lstm.h5')
         with open(pkl_filename,'wb') as file:
             pickle.dump(lstm_model,file)
    y_pred = np.around(y_pred)
    result = cohen_kappa_score(y_test.values,y_pred,weights='quadratic')
    print("Kappa Score: {}".format(result))
    results.append(result)

    count += 1


# In[ ]:




