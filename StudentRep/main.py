# Imports:
import pandas as pd
import numpy as np
import pickle
# sklearn:
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
# Matplotlib:
import matplotlib.pyplot as pyplot
from matplotlib import style

# File paths:
DATA_PATH = '../Dataset/student-mat.csv'
SAVE_PATH = '../SavedModels/studentmodel.pickle'

data = pd.read_csv(DATA_PATH, sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Uncomment to train/calculate models:
"""
best_accuracy = 0
accuracy = 0
best_run_number = -1

for run in range(50):
    # Run 30 times and keep the best model that was generated.

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    accuracy = linear.score(x_test, y_test)
    print(f'Accuracy for run {run + 1}: {round(accuracy, 2)}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_run_number = run
        with open(SAVE_PATH, 'wb') as model_save_file:
            # Save the generated model
            pickle.dump(linear, model_save_file)


print(f"The best model was generated in run {best_run_number} with the accuracy {round(best_accuracy, 2)}")
"""

pickle_in = open(SAVE_PATH, 'rb')
linear = pickle.load(pickle_in)

"""
print(f"Coeficient: {linear.coef_}")
print(f"Intercept: {linear.intercept_}")
"""

predictions = linear.predict(x_test)


for x in range(len(predictions)):
    res = round(predictions[x])
    close = 'no'
    if abs(y_test[x] - res) <= 2:
        close = 'yes'

    print(round(predictions[x]), x_test[x], y_test[x], f"Accurate? {close}")

style.use("ggplot")
x = 'absences' # This will be X for the graph <-- this is the 'comparison' between G3 and G1 or G2
y = 'G3' # This will be Y for the graph
pyplot.scatter(data[y], data[x])
pyplot.xlabel(f'Comparsion between the {x} grade and the {y} grade')
pyplot.ylabel(f'Compared grade ({y})')
pyplot.show()
