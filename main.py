#@author Nicholas Haynes

#imports
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 


#Loading in our data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names = names)


# This gives an idea of how many instances(rows) and how many attributes (cols) the data contains, it is commented out because I don't need to know everytime I run this program
# print(dataset.shape)

# This lets me see the data directly, again it is commented out because it is only useful sometimes but since I am learning I would like to refer back to it.
# print(dataset.head(20))

#Some descriptions and more information
# print(dataset.describe())

# Some box and whisker plots
# dataset.plot(kind='box', subplots = True, layout = (2,2), sharex=False, sharey = False)
# plt.show()

#Some histograms
# dataset.hist()
# plt.show()

#scatter plots
# scatter_matrix(dataset)
# plt.show()

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validationSize = 0.20
seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size = validationSize, random_state = seed)

# Test options
seed = 7
scoring = 'accuracy'


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)