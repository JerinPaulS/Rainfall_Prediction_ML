import keras
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils, to_categorical
from numpy import argmax
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv(r'G:\BE Project\Dataset\Final-dataset.csv')


dataset["Date"] = dataset.Date.convert_objects(convert_numeric=True)
dataset["Prediction"] = dataset['Prediction'].astype(int)

imputer = SimpleImputer(strategy = 'mean')
dataset = pd.DataFrame(imputer.fit_transform(dataset))

atm_parameters = dataset.iloc[:,1:-1].values
rainfall_result = dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(atm_parameters, rainfall_result, test_size=0.2, shuffle= True)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

fig = plt.figure()

from sklearn.metrics import classification_report
print('Classification Report: \n\n{}\n\n'.format(classification_report(y_pred, y_test)))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))

def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)

visualize_classifier(DecisionTreeClassifier(), X_test, y_test)
plt.show()
