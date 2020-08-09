#Importing all the required libraries:

import csv
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


#Reading Dataset:

data = pd.read_csv("/data/training/book_reviews.csv", names = ["Text", "Label"])

#Output1:

row = data.shape
o1 = pd.DataFrame(row)
o1.drop([1], inplace = True)
o1.to_csv("/code/output/output1.csv", index = False, header = False)

#Tokenizing text:

def tokenize(text): 
    tk = TweetTokenizer()
    return tk.tokenize(text)
countVectorizer = CountVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,ngram_range=(1, 1))
count = countVectorizer.fit_transform(data['Text'])
num_unigrams = count.shape[1]

#Output2:

col = ["a"]
o4 = pd.DataFrame(columns = col)
o4.loc[0, "a"] = num_unigrams
o4.to_csv("/code/output/output2.csv", index = False, header = False)

#Label Encoding for Label column:

label_encoder = preprocessing.LabelEncoder()

data["Label"] = label_encoder.fit_transform(data["Label"])
data["Label"].unique()

#Splitting the Dataset:

x = data['Text']
y = data['Label']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=100, test_size=0.3)
vectorizer = TfidfVectorizer(max_features=1000)
x_train_idf = vectorizer.fit_transform(x_train)
x_test_idf = vectorizer.transform(x_test)
df_idf = pd.DataFrame(np.round(vectorizer.idf_,3), index=vectorizer.get_feature_names(),columns=["idf_weights"])

#Output3:

df_idf.sort_values(by=['idf_weights'],ascending = False).head().to_csv("/code/output/output3.csv", index=False, header=False)

#Naive Bayes Classifier:

nb = MultinomialNB()
nb.fit(x_train_idf, y_train)

y_pred = nb.predict(x_test_idf)
score = accuracy_score(y_test, y_pred)

#Output4:

col = ["a"]
o4 = pd.DataFrame(columns = col)
o4.loc[0, "a"] = score
o4.to_csv("/code/output/output4.csv", index = False, header = False)

#Output5:
cm = confusion_matrix(y_test, y_pred)
o5 = pd.DataFrame(cm)
o5 = o5.transpose()
o5.to_csv("/code/output/output5.csv", index = False, header = False)