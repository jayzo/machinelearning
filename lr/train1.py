import numpy as np
import urllib.request
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
# url with dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
raw_data = urllib.request.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:8]
y = dataset[:,8]
model = LogisticRegression()
model.fit(X, y)
print(model)
joblib.dump(model, 'train1.pkl')