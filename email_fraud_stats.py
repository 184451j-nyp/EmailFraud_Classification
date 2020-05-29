from statistics import mean

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

import utility as ut

dataframe = ut.get_clean_dataframe('extracted_emails.csv')
x, y = dataframe['content'], dataframe['target']

x = x.apply(lambda string: ut.stem_string(string))
vectorizer = TfidfVectorizer()
clf = LinearSVC()
skf = StratifiedKFold(shuffle=True, random_state=42)
accuracy_arr = []
precision_arr = []
f1_arr = []
for train_index, test_index in skf.split(x, y):
	x_train, x_test = x.iloc[train_index], x.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	x_train = vectorizer.fit_transform(x_train)
	x_test = vectorizer.transform(x_test)
	clf.fit(x_train, y_train)

	conf_mat = plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues, normalize='true')
	conf_mat.ax_.set_title('Confusion matrix with {} samples'.format(len(y_test)))
	plt.show()
	y_pred = clf.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	accuracy_arr.append(accuracy)
	precision = precision_score(y_test, y_pred, labels=['spam', 'ham'], pos_label='spam')
	precision_arr.append(precision)
	f1 = f1_score(y_test, y_pred, labels=['spam', 'ham'], pos_label='spam')
	f1_arr.append(f1)

	print(f'\nAccuracy: {accuracy}')
	print(f'Precision: {precision}')
	print(f'F1: {f1}')

print(f'\nMean accuracy: {mean(accuracy_arr) * 100}')
print(f'Mean precision: {mean(precision_arr) * 100}')
print(f'Mean F1: {mean(f1_arr) * 100}')
