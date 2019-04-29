import keras
import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_csv(r'D:\BE Project\Final-dataset11plot.csv')

dataset["Date"] = dataset.Date.convert_objects(convert_numeric=True)
dataset["Prediction"] = dataset['Prediction'].astype(int)

imputer = SimpleImputer(strategy = 'mean')
dataset = pd.DataFrame(imputer.fit_transform(dataset))

atm_parameters = dataset.iloc[:,1:-1].values
rainfall_result = dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(atm_parameters, rainfall_result, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = linear_model.LinearRegression()
results_formula = classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = atm_parameters[:,0]
x2 = atm_parameters[:,3]
x3 = atm_parameters[:,4]
x4 = atm_parameters[:,5]
x5 = atm_parameters[:,6]

ax.set_xlabel('Temperature')
ax.set_ylabel('Pressure')
ax.set_zlabel('Humidity')

x_surf, y_surf = np.meshgrid(np.linspace(x1.min(), x1.max(), 40), np.linspace(x2.min(), x2.max(), 40))
fittedY = classifier.predict(X_test)
ax.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape), color='None', alpha=0.01)
ax.scatter3D(x1, x2, x3, c=x3, cmap='Greens')

#ax.scatter3D(x1, x2, x3, c=x3, cmap='Greens');
#ax.scatter3D(x3, x4, x5, c=x3, cmap='Greens');

plt.show()
