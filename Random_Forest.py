import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.optimizers import adam, sgd, rmsprop
from sklearn.ensemble import RandomForestClassifier
from keras.utils import np_utils, to_categorical
from numpy import argmax
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv(r'D:\BE Project\Final-dataset11plot.csv')


dataset["Date"] = dataset.Date.convert_objects(convert_numeric=True)
dataset["Prediction"] = dataset['Prediction'].astype(int)

imputer = SimpleImputer(strategy = 'mean')
dataset = pd.DataFrame(imputer.fit_transform(dataset))

atm_parameters = dataset.iloc[:,1:-1].values
rainfall_result = dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(atm_parameters, rainfall_result, test_size=0.5, shuffle= True)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#y_train = np_utils.to_categorical(y_train, 73)
#y_test = np_utils.to_categorical(y_test, 73)


classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)



#y_pred = (np.argmax(y_pred, axis=1)).reshape(-1, 1)
#y_test = (np.argmax(y_test, axis=1)).reshape(-1, 1)


from sklearn.metrics import classification_report
print('Classification Report: \n\n{}\n\n'.format(classification_report(y_pred, y_test)))

