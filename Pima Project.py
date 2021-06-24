# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:01:07 2021

@author: Abduallah_Gamal
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv('C:\\Users\\Abduallah_Gamal\\Desktop\\archive\\pima.csv')
# print(df.head(20))
# print(df.info())

#========================================================================================
# sns.countplot(x='Outcome',data=df, palette='hls')
# sns.countplot(x='Glucose',data=df, palette='hls')
# sns.countplot(x='Pregnancies',data=df, palette='hls')

# sns.heatmap(df.corr())
# sns.pairplot(df, hue="Outcome")
# sns.pairplot(df, vars=["Pregnancies", "BMI"])

# g = sns.FacetGrid(df, row="Pregnancies", col="Outcome", margin_titles=True)
# bins = np.linspace(0, 50, 20)
# g.map(plt.hist, "BMI", color="steelblue", bins=bins, lw=0)

#========================================================================================

columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
            'BMI','DiabetesPedigreeFunction','Age']
labels = df['Outcome'].values
features = df[list(columns)].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)
# clf = RandomForestClassifier(n_estimators=100)
# clf = clf.fit(X_train, y_train)


# accuracy = clf.score(X_train, y_train)
# print (' اداء النموذج في عينة التدريب بدقة ', accuracy*100)

# accuracy = clf.score(X_test, y_test)
# print (' اداء النموذج في عينة الفحص بدقة ', accuracy*100)

# ypredict = clf.predict(X_train)
# print('\n Training classification report\n', classification_report(y_train, ypredict))
# print("\n Confusion matrix of training \n", confusion_matrix(y_train, ypredict))

# ypredict = clf.predict(X_test)
# print( '\n Testing classification report\n', classification_report(y_test, ypredict))
# print( "\n Confusion matrix of Testing \n", confusion_matrix(y_test, ypredict))

#========================================================================================
# تجربة تحسين اداء النموذج باستخدام طريقة scaler
# scaling
# scaler = StandardScaler()

# # Fit only on training data
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# # apply same transformation to test data
# X_test = scaler.transform(X_test)

# clf = RandomForestClassifier(n_estimators=100)
# clf = clf.fit(X_train, y_train)


# accuracy = clf.score(X_train, y_train)
# print (' اداء النموذج في عينة التدريب بدقة ', accuracy*100)

# accuracy = clf.score(X_test, y_test)
# print (' اداء النموذج في عينة الفحص بدقة ', accuracy*100)

# ypredict = clf.predict(X_train)
# print('\n Training classification report\n', classification_report(y_train, ypredict))
# print("\n Confusion matrix of training \n", confusion_matrix(y_train, ypredict))

# ypredict = clf.predict(X_test)
# print( '\n Testing classification report\n', classification_report(y_test, ypredict))
# print( "\n Confusion matrix of Testing \n", confusion_matrix(y_test, ypredict))



# ================================================================================================
# تجربة تحسين اداء النموذج بطريقةmin-max scaler

scaler = preprocessing.MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=5)
clf = clf.fit(X_train, y_train)

accuracy = clf.score(X_train, y_train)
print(' اداء النموذج في عينة التدريب بدقة ', accuracy*100)

accuracy = clf.score(X_test, y_test)
print(' اداء النموذج في عينة الفحص بدقة ', accuracy*100)

ypredict = clf.predict(X_train)
print('\n Training classification report\n', classification_report(y_train, ypredict))
print( "\n Confusion matrix of training \n", confusion_matrix(y_train, ypredict))

ypredict = clf.predict(X_test)
print('\n Testing classification report\n', classification_report(y_test, ypredict))
print("\n Confusion matrix of Testing \n", confusion_matrix(y_test, ypredict))






