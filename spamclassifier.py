# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:44:20 2021

@author: Nishant Mishra
"""

import pandas as pd

messages = pd.read_csv(r'D:/DATA SCIENCE DATA SETS/NLP/SMS Spam Collection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])

import nltk
nltk.download()
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
lemmatizer=WordNetLemmatizer()

corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

