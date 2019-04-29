import keras
import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'D:\BE Project\Final-dataset.csv')

dataset["Date"] = dataset.Date.convert_objects(convert_numeric=True)
dataset["Prediction"] = dataset['Prediction'].astype(int)

imputer = SimpleImputer(strategy = 'mean')
dataset = pd.DataFrame(imputer.fit_transform(dataset))

atm_parameters = dataset.iloc[:, 1:-1].values
rainfall_result = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(atm_parameters, rainfall_result, test_size=0.2)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


'''
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
'''

from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()

classifier.add(Dense(100, kernel_initializer='uniform', activation='relu', input_shape=(9,)))
classifier.add(Dropout(0.2))
classifier.add(Dense(100, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(100, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(100, kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(100, kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(100, kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(100, kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(1, kernel_initializer='uniform'))

classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

classifier.fit(X_train,y_train, batch_size=100, epochs=100, validation_split=0.1)

y_pred = classifier.predict(np.asarray([28, 33, 23, 1014.2, 69, 0, 8.9, 1.9, 7.6]).reshape(1,9))
print(y_pred)