#!/usr/bin/env python
# coding: utf-8

# In[2]:

import warnings
warnings.filterwarnings('ignore')

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



# In[ ]:
def train():    
    sent_list = []
    results=[]
    
    cv = KFold(5, shuffle=True)
    
    count=1
    
    pkl_filename="nlp_model.pkl"
    pkl1_filename="nlp1_model.pkl"
    
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
             with open(pkl1_filename,'wb') as file:
                 pickle.dump(model,file)
                 
        y_pred = np.around(y_pred)
        result = cohen_kappa_score(y_test.values,y_pred,weights='quadratic')
        print("Kappa Score: {}".format(result))
        results.append(result)
    
        count += 1
        break
    return results

def get_predictions(text_input):
    
    
    with open("nlp1_model.pkl",'rb') as file:
        w2vec_model = pickle.load(file)
    
    w2vec_model.wv.save_word2vec_format('word2vecmodel.bin', binary=True)
    
    with open('nlp_model.pkl','rb') as file2:
        model = pickle.load(file2)
#    print(model.summary())
#    print()
#    sent_list = sentence(text_input)
#    print(sent_list)
    
#    essay_test_words = []
#    
    essay_test_words = word(text_input)
        
    input_vec = [word_vectors(essay_test_words ,w2vec_model,300)]
#    print(input_vec)
    input_arr_vec = np.array(input_vec)
    input_arr_vec = np.reshape(input_arr_vec, (input_arr_vec.shape[0], 1, input_arr_vec.shape[1]))
    y_pred = model.predict(input_arr_vec)
    y_pred = np.around(y_pred)

    return y_pred[0][0]

    
#if __name__ == "__main__":
#    print(get_predictions("Dear local newspaper, I think effects computers have on people are great learning skills/affects because they give us time to chat with friends/new people, helps us learn about the globe(astronomy) and keeps us out of troble! Thing about! Dont you think so? How would you feel if your teenager is always on the phone with friends! Do you ever time to chat with your friends or buisness partner about things. Well now - there's a new way to chat the computer, theirs plenty of sites on the internet to do so: @ORGANIZATION1, @ORGANIZATION2, @CAPS1, facebook, myspace ect. Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it. How did you learn about other countrys/states outside of yours? Well I have by computer/internet, it's a new way to learn about what going on in our time! You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1's you'll be surprise at how much he/she knows. Believe it or not the computer is much interesting then in class all day reading out of books. If your child is home on your computer or at a local library, it's better than being out with friends being fresh, or being perpressured to doing something they know isnt right. You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by. Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place. Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble. Thank you for listening."))
#    predict()
