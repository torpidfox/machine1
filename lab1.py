import sklearn.naive_bayes as nb
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
#from sklearn.preprocessing.imputation import Imputer
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

E1 = [10, 14]
D1 = [[16, 0], [0, 16]]

E2 = [20, 18]
D2 = [[9, 0], [0, 9]]
count = 50

def prepare_dataset(filename, to_drop=None, to_factorize=None, to_fill=None):
	with open(filename) as f:
		dataset = pd.read_csv(f, sep=',')

	dataset = dataset.drop(to_drop, axis=1)

	#convert string features to numbers
	for column in to_factorize:
		dataset[column] = pd.factorize(dataset[column])[0]

	imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	#imp = Imputer(missing_values=np.nan, strategy='mean')

	for column in to_fill:
		column_reshaped = [[el] for el in dataset[column]]
		dataset[column] = imp.fit(column_reshaped).transform([[el] for el in dataset[column]])

	return dataset

def evaluate(data, sizes, clf_constructor):
	acc = list()
	dataset = shuffle(data)

	for i in train_size:
		clf = clf_constructor()
		clf.fit(dataset[dataset.columns[:-1]][:i],
			dataset[dataset.columns[-1]][:i])
		acc.append(clf.score(dataset[dataset.columns[:-1]][i:], dataset[dataset.columns[-1]][i:]))

	return acc

def task2():
	features = np.append(np.random.multivariate_normal(E1, D1, size=[count]),
		np.random.multivariate_normal(E2, D2, size=[count]),
		axis=0)

	data = {'features' : features,
	'tags' : [0] * count + [1] * count}

	#classify
	train_size = [1, 10, 40]
	clf = nb.GaussianNB()
	clf.fit(data['features'], data['tags'])
	predictions = clf.predict(data['features'])
	accuracy = accuracy_score(data['tags'], predictions)
	print(accuracy)

	points_x, points_y = zip(*data['features'][:count])
	plt.plot(points_x, points_y, 'o')
	points_x, points_y = zip(*data['features'][count:])
	plt.plot(points_x, points_y, 'ro')
	plt.show()


def plot(data, xlabel=None, ylabel=None, title=None, xvals=None):
	plt.plot(data)
	
	if xvals:
		plt.xticks(list(range(len(data))), xvals)
	if xlabel:
		plt.xlabel(xlabel)
	if ylabel:
		plt.ylabel(ylabel)
	if  title:
		plt.title(title)
	
	plt.show()


#read train data
dataset = prepare_dataset('../Titanic_train.csv',
	to_drop=['PassengerId', 'Cabin', 'Ticket'],
	to_factorize=['Sex', 'Embarked'],
	to_fill=['Age'])

clf = nb.GaussianNB()
clf.fit(dataset.drop('Survived', axis=1).values[:350], 
	list(dataset['Survived'])[:350])

acc = clf.score(dataset.drop('Survived', axis=1).values[350:], 
	list(dataset['Survived'])[350:])

print(acc)


#read test data
dataset = prepare_dataset('../Titanic_test.csv',
	to_drop=['PassengerId', 'Cabin', 'Ticket'],
	to_factorize=['Sex', 'Embarked'],
	to_fill=['Age'])

dataset = dataset.dropna()

train_size = [10, 150, 300]
print(clf.predict(dataset.values))
# acc = evaluate(dataset, train_size, nb.GaussianNB)
# plot(acc, 
# 	xlabel='Размер тренировочной выборки',
# 	ylabel='Точность на валидационной выборке',
# 	title='Зависимость качества обучения от объема тренировочной выборки',
# 	xvals=train_size)

#spam dataset

# with open('../spambase/spambase.data') as f:
# 	dataset = pd.read_csv(f, header=None)

# train_size = [10, 50, 100, 1000, 3000, 3500, 4000]
# acc = evaluate(dataset, train_size, nb.GaussianNB)
# plot(acc, 
# 	xlabel='Размер тренировочной выборки',
# 	ylabel='Точность на валидационной выборке',
# 	title='Зависимость качества обучения от объема тренировочной выборки',
# 	xvals=train_size)

# #tictac dataset

# with open('../Tic_tac_toe.txt') as f:
# 	dataset = pd.read_csv(f, header=None)

# dataset = dataset.stack().rank(method='dense').unstack()
# train_size = [10, 50, 100, 500, 800]
# acc = evaluate(dataset, train_size, nb.MultinomialNB)
# plot(acc, 
# 	xlabel='Размер тренировочной выборки',
# 	ylabel='Точность на валидационной выборке',
# 	title='Зависимость качества обучения от объема тренировочной выборки',
# 	xvals=train_size)