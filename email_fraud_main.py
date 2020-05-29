from os.path import isfile
from joblib import load
import nltk
import utility as ut

model_filename = 'email_fraud_model.pickle'
csv_filename = 'extracted_emails.csv'

# noinspection SpellCheckingInspection
def main():
	if not isfile(model_filename):
		print('Waiting for model to load...')
		ut.train_model(csv_filename, model_filename=model_filename)
	clf = load(model_filename)
	vectorizer = clf['vectorizer']
	guess_str = ut.stem_string(input('Enter text to see if it is spam: '))
	prediction = vectorizer.transform([guess_str])
	result = clf['classifier'].predict(prediction)
	print(f'This text is {result[0]}')


if __name__ == '__main__':
	for name in ('tokenizers/punkt', 'corpora/stopwords'):
		try:
			nltk.data.find(name)
		except LookupError:
			file = name.split('/')[1]
			nltk.download(file)
	main()
