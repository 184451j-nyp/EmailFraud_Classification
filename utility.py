import numpy as np
import pandas as pd
import string
from joblib import dump
import nltk
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def stem_string(input):
	stemmer = SnowballStemmer('english', ignore_stopwords=True)
	tokens = nltk.word_tokenize(input.replace('.', '. ').replace(',', ', ').replace('--', ''))
	stemmed_arr = [stemmer.stem(word) for word in tokens if word not in string.punctuation]
	return ' '.join(stemmed_arr)

# noinspection SpellCheckingInspection
def train_model(csv_filename, model_filename=None):
	dataframe = get_clean_dataframe(csv_filename)
	X_train_dataset, y_train_dataset = dataframe['content'], dataframe['target']

	X_train_dataset = X_train_dataset.apply(lambda string: stem_string(string))
	vectorizer = TfidfVectorizer()
	X_train_dataset = vectorizer.fit_transform(X_train_dataset)
	clf = LinearSVC()
	clf.fit(X_train_dataset, y_train_dataset)
	if model_filename is None:
		return {'vectorizer': vectorizer, 'classifier': clf}
	else:
		dump({'vectorizer': vectorizer, 'classifier': clf}, model_filename)

def get_clean_dataframe(csv_filename):
	dataframe = pd.read_csv(csv_filename, delimiter='|')
	# data cleaning
	# dataframe.replace('<.*?>|(&nbsp;)|=\d\w|\\\w*', '', inplace=True, regex=True)
	dataframe.replace('\n|\t', ' ', inplace=True, regex=True)
	dataframe.replace('', np.nan, inplace=True)
	dataframe.dropna(inplace=True)
	return dataframe
