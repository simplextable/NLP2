import pandas as pd

train = pd.read_csv('train.tsv', sep="\t")

train.head(10)

train.groupby("Sentiment").count()

train["Sentiment"].replace(0, value = "negatif" , inplace = True)
train["Sentiment"].replace(1, value = "negatif" , inplace = True)

train["Sentiment"].replace(3, value = "pozitif" , inplace = True)
train["Sentiment"].replace(4, value = "pozitif" , inplace = True)

train = train[(train.Sentiment == "negatif") | (train.Sentiment == "pozitif")]

df = pd.DataFrame()
df["Phrase"] = train["Phrase"]
df["sonuç"] = train["Sentiment"]

df.head()

#Lower Case
df['Phrase'] = df['Phrase'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#Punctuation
df['Phrase'] = df['Phrase'].str.replace('[^\w\s]','')

#Digits
df['Phrase'] = df['Phrase'].str.replace('\d','')

df.head()

#Stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['Phrase'] = df['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

df.head()

#Deleting Less Words
sil = pd.Series(' '.join(df['Phrase']).split()).value_counts()[-1000:]
df['Phrase'] = df['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

#Lemmization
from textblob import Word
nltk.download('wordnet')
df['Phrase'] = df['Phrase'].apply(lambda x: " ".join([Word(i).lemmatize() for i in x.split()]))

df.head()

df.info()

from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
df["sonuç"] = encoder.fit_transform(df["sonuç"])

df.head()

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(df["Phrase"],df["sonuç"],  random_state = 1)

"""**Count Vectors**"""

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
vec.fit(train_x)

x_train_count = vec.transform(train_x)
x_test_count = vec.transform(test_x)

x_train_count.toarray()

"""# TF-IDF"""

#wordlevel

from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)

x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

# ngram level tf-idf

tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range = (2,3))
tf_idf_ngram_vectorizer.fit(train_x)

x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(train_x)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)

# characters level tf-idf

tf_idf_chars_vectorizer = TfidfVectorizer(analyzer = "char", ngram_range = (2,3))
tf_idf_chars_vectorizer.fit(train_x)

"""**Chars**"""

tf_idf_chars_vectorizer = TfidfVectorizer(analyzer = "char", ngram_range = (2,3))
tf_idf_chars_vectorizer.fit(train_x)

x_train_tf_idf_chars = tf_idf_chars_vectorizer.transform(train_x)
x_test_tf_idf_chars = tf_idf_chars_vectorizer.transform(test_x)

"""**Models**"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
nb = MultinomialNB()
nb_model = nb.fit(x_train_count,train_y)
accuracy = cross_val_score(nb_model, 
                                 x_test_count, 
                                 test_y, 
                                 cv = 10).mean()

print("Count Vectors Accurancy Rate:", accuracy)

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
nb = MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_word,train_y)
accuracy = cross_val_score(nb_model, 
                                 x_test_tf_idf_word, 
                                 test_y, 
                                 cv = 10).mean()

print("TF-Idf Word Level Accurancy Rate:", accuracy)

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
nb = MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_ngram,train_y)
accuracy = cross_val_score(nb_model, 
                                 x_test_tf_idf_ngram, 
                                 test_y, 
                                 cv = 10).mean()

print("TF-Idf N-Gram Accurancy Rate:", accuracy)

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
nb = MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_chars,train_y)
accuracy = cross_val_score(nb_model, 
                                 x_test_tf_idf_chars, 
                                 test_y, 
                                 cv = 10).mean()

print("TF-Idf Chars Accurancy Rate:", accuracy)

import xgboost
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_count,train_y)
accuracy = cross_val_score(xgb_model, 
                                   x_test_count, 
                                   test_y, 
                                   cv = 10).mean()

print("Count Vectors Accurancy Rate:", accuracy)

import xgboost
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_word,train_y)
accuracy = cross_val_score(xgb_model, 
                                   x_test_tf_idf_word, 
                                   test_y, 
                                   cv = 10).mean()

print("Tf-İdf Word Level Accurancy Rate:", accuracy)