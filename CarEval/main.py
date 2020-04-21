# Imports:
import pandas as pd
import numpy as np
# sklearn:
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

DATA_PATH = '../Dataset/car.data'

data = pd.read_csv(DATA_PATH)
#print(data.head())

# Convert to numerical:
le = preprocessing.LabelEncoder()
buying = le.fit_transform((data['buying']))
maint = le.fit_transform((data['maint']))
door = le.fit_transform((data['door']))
persons = le.fit_transform((data['persons']))
lug_boot = le.fit_transform((data['lug_boot']))
safety = le.fit_transform((data['safety']))
class_ = le.fit_transform((data['class']))

predict = 'class'

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(class_)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood'] # Recoverting numbers into string so it's easyer to see

for _ in range(len(predicted)):
    print(f'Pred: {names[predicted[_]]} | Data: {x_test[_]} | Acctual: {names[y_test[_]]}')
    # knbgs = model.kneighbors([x_test[_]], 7, True)
    # print("Neighbours: ", knbgs)
