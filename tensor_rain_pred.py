import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.optimizers import adam, sgd, rmsprop
from keras.utils import np_utils, to_categorical
from numpy import argmax
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv(r'D:\BE Project\Final-dataset.csv')

dataset["Date"] = dataset.Date.convert_objects(convert_numeric=True)
dataset["Prediction"] = dataset['Prediction'].astype(int)
df = pd.read_csv(r'D:\BE Project\Final-dataset.csv')
y = df.iloc[:,6]
x = df.iloc[:,10]


imputer = SimpleImputer(strategy = 'mean')
dataset = pd.DataFrame(imputer.fit_transform(dataset))

row=0
for row in range(11688):
    if x(row) < 10:
        print('hello')

atm_parameters = dataset.iloc[:,1:-1].values
rainfall_result = dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(atm_parameters, rainfall_result, test_size=0.2, shuffle= True)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


y_train = np_utils.to_categorical(y_train, 73)
y_test = np_utils.to_categorical(y_test, 73)


from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()  # type: Sequential

classifier.add(Dense(100,kernel_initializer='uniform',activation='relu',input_shape=(9,)))
classifier.add(Dense(100,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(100,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(73,kernel_initializer='uniform',activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X_train,y_train, batch_size=100,epochs=20,validation_split=0.1,shuffle=True)

y_pred = classifier.predict(X_test)

y_pred = (np.argmax(y_pred, axis=1)).reshape(-1, 1)
y_train = (np.argmax(y_train, axis=1)).reshape(-1, 1)
from sklearn.metrics import classification_report
print('Classification Report: \n\n{}\n\n'.format(classification_report(y_pred, y_train)))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test,y_pred)
# print(cm)


res = 0

# for i in range(2338):
#     res = res + abs(y_pred[i]-y_test[i])
#
# print(res/2338)