import sklearn.naive_bayes as nb
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

E1 = [10, 14]
D1 = [[4, 0], [0, 4]]

E2 = [20, 18]
D2 = [[3, 0], [0, 3]]
count = 50

def prepare_dataset(filename, to_drop=None, to_factorize=None, to_fill=None):
	with open(filename) as f:
		dataset = pd.read_csv(f, sep=',')

	dataset = dataset.drop(to_drop, axis=1)

	#convert string features to numbers
	for column in to_factorize:
		dataset[column] = pd.factorize(dataset[column])[0]

	imp = SimpleImputer(missing_values=np.nan, strategy='mean')

	for column in to_fill:
		column_reshaped = [[el] for el in dataset[column]]
		dataset[column] = imp.fit(column_reshaped).transform([[el] for el in dataset[column]])

	return dataset




#generate data
features = np.append(np.random.multivariate_normal(E1, D1, size=[count]),
	np.random.multivariate_normal(E2, D2, size=[count]),
	axis=0)

data = {'features' : features,
'tags' : [0] * count + [1] * count}

#classify
clf = nb.MultinomialNB()
clf.fit(data['features'], data['tags'])
predictions = clf.predict(data['features'])
accuracy = accuracy_score(data['tags'], predictions)
print(accuracy)

points_x, points_y = zip(*data['features'][:count])
plt.plot(points_x, points_y, 'o')
points_x, points_y = zip(*data['features'][count:])
plt.plot(points_x, points_y, 'ro')
plt.show()


#read train data
dataset = prepare_dataset('Titanic_train.csv',
	to_drop=['PassengerId', 'Cabin', 'Ticket'],
	to_factorize=['Sex', 'Embarked'],
	to_fill=['Age'])

clf = nb.GaussianNB()
clf.fit(dataset.drop('Survived', axis=1).as_matrix(), 
	list(dataset['Survived']))


#read test data
dataset = prepare_dataset('Titanic_test.csv',
	to_drop=['PassengerId', 'Cabin', 'Ticket'],
	to_factorize=['Sex', 'Embarked'],
	to_fill=['Age'])

# print(np.where(data.values==float('nan')))

# pred = clf.predict(dataset.values)
# print(pred)

#read spam dataset

with open('spambase/spambase.data') as f:
	dataset = pd.read_csv(f, header=None)

print(len(dataset))
acc = list()
train_size = [500, 1000, 3000, 3500, 4000]
dataset = shuffle(dataset)

for i in train_size:
	clf = nb.GaussianNB()
	clf.fit(dataset[dataset.columns[:-1]][:i],
		dataset[dataset.columns[-1]][:i])
	shuffled = dataset
	acc.append(clf.score(shuffled[shuffled.columns[:-1]][4000:], shuffled[shuffled.columns[-1]][4000:]))

print(acc)

with open('Tic_tac_toe.txt') as f:
	dataset = pd.read_csv(f, header=None)


for i in list(dataset):
	dataset[i] = pd.factorize(dataset[i])[0]

train_size = [10, 50, 100, 500, 800]
dataset = shuffle(dataset)
acc = list()
for i in train_size:
	clf = nb.GaussianNB()
	clf.fit(dataset[dataset.columns[:-1]][:i],
		dataset[dataset.columns[-1]][:i])
	shuffled = dataset
	acc.append(clf.score(shuffled[shuffled.columns[:-1]][800:], shuffled[shuffled.columns[-1]][800:]))

# print(acc)

# plt.plot(acc)
# plt.xticks(list(range(len(acc))), train_size)
# plt.xlabel('Размер обучающей выборки')
# plt.ylabel('Точность классификатора на тестовой выборке')
# plt.title('Tic tac toe dataset')
# plt.show()

#read iris dataset








