# sklearn:
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

data = datasets.load_breast_cancer()

# print(data.feature_names)
# print(data.target_names)

x = data.data
y = data.target

names = ['malignant' 'benign']
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=5)  # KNN
# clf = svm.SVC(kernel='linear', C=2) # C- soft margin  # SVM

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)
