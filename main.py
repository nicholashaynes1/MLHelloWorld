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