import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)

lr = LogisticRegression(C=100.0, random_state = 1, solver = 'lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

svm = SVC(kernel='linear',C=1.0, random_state=1, probability=True)
svm.fit(X_train_std, y_train)

tree_model = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
tree_model.fit(X_train,y_train)

knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)


with open('../models/lr.pck', 'wb') as f:
    pickle.dump((sc, lr), f)

with open('../models/svm.pck', 'wb') as f:
    pickle.dump((sc, svm), f)

with open('../models/tree_model.pck', 'wb') as f:
    pickle.dump((sc, tree_model), f)

with open('../models/knn.pck', 'wb') as f:
    pickle.dump((sc, knn), f)

