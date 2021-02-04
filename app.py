import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
import re


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,roc_auc_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 

import tensorflow as tf


from math import ceil
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
# from prettytable import PrettyTable
from keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, LSTM,Dropout, concatenate,\
Conv1D,BatchNormalization,SpatialDropout1D,Bidirectional,GRU,GlobalAveragePooling1D,GlobalMaxPooling1D
from tensorflow.keras import regularizers,initializers,optimizers,Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint,EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer



data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

app = Flask(__name__)
model = joblib.load('logistic.pkl')
print('model_loaded')


@app.route('/')
def home():
    return render_template('ind.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method=='POST':
        text = request.form.get('comment')
    def clean_text(text):
        text = str(text)
        text = text.lower()
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)
        text = re.sub('\W', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub('[0-9]', '', text)
        text = text.strip(' ')
        return text       
        
    text = clean_text(text)
    
    
    data['comment_text'] = data['comment_text'].map(lambda com : clean_text(com))
    X = data['comment_text']
    vect = TfidfVectorizer(max_features=5000,stop_words='english')
    X_dtm = vect.fit_transform(X)
    

    X_test = text
    Xvdm_test = vect.transform([X_test])   
    
    
    
    logreg = LogisticRegression(C=10)
    test_y_prob = []
    labels=[]
    target = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    for label in target:
        print('... Processing {}'.format(label))
        y = data[label]
        # train the model using X_dtm & y
        logreg.fit(X_dtm, y)
        
        # compute the training accuracy
        y_pred_X = logreg.predict(Xvdm_test)
        test_y_prob.append(logreg.predict(Xvdm_test))
        labels.append(label)
    a = {}
    for i in range(len(labels)):
        a[labels[i]]=test_y_prob[i].tolist()
        
    
        
    return render_template('ind.html', prediction_text='Results are: \n {}'.format(a))





#     return render_template('ind.html', prediction_text='Toxicity type of {} is {} '.format(labels,test_y_prob))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#         text = request.args.get('how are you')
# #     text =request.args.get("text")
#     def clean_text(text):
#         text = str(text)
#         text = text.lower()
#         text = re.sub(r"what's", "what is ", text)
#         text = re.sub(r"\'s", " ", text)
#         text = re.sub(r"\'ve", " have ", text)
#         text = re.sub(r"can't", "cannot ", text)
#         text = re.sub(r"n't", " not ", text)
#         text = re.sub(r"i'm", "i am ", text)
#         text = re.sub(r"\'re", " are ", text)
#         text = re.sub(r"\'d", " would ", text)
#         text = re.sub(r"\'ll", " will ", text)
#         text = re.sub(r"\'scuse", " excuse ", text)
#         text = re.sub('\W', ' ', text)
#         text = re.sub('\s+', ' ', text)
#         text = re.sub('[0-9]', '', text)
#         text = text.strip(' ')
#         return text
    
    
#     data['comment_text'] = data['comment_text'].map(lambda com : clean_text(com))
#     X = data['comment_text']
#     vect = TfidfVectorizer(max_features=5000,stop_words='english')
#     X_dtm = vect.fit_transform(X)
    
# #     test['comment_text'] = test['comment_text'].map(lambda x: clean_text(x))
#     text = clean_text(text)
# #     X_test = text
# #     Xvdm_test = vect.transform([X_test])
    
    
# #     logreg = LogisticRegression(C=10)
# #     test_y_prob = []
# #     labels=[]
# #     target = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
# #     for label in target:
# #         print('... Processing {}'.format(label))
# #         y = data[label]
# #         # train the model using X_dtm & y
# #         logreg.fit(X_dtm, y)
        
# #         # compute the training accuracy
# #         y_pred_X = logreg.predict(Xvdm_test)
# #         test_y_prob.append(logreg.predict_proba(Xvdm_test))
# #         labels.append(label)
     