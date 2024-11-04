# Lab1 part 1 bla bla bla

# Dependencies - Importing all necessary Python Packages
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


# Load the train and test datasets to create two DataFrames ()
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

# Preview data by printing it out (middle rows shows up as ...?)
print("***** Train_Set *****")
print(train.head())
print("\n")
print("***** Test_Set *****")
print(test.head())

# Intial statistics of train
print("\n***** Train_Set *****")
print(train.describe())
print("\n***** Test_Set *****")
print(test.describe())

# Print out name of columns
print("\n***** Columns *****")
print(train.columns.values)

# Find out missing values
# For the train set
train.isna().head()
# For the test set
test.isna().head()
## Print out total number of missing values
# print("\n*****In the train set*****")
# print(train.isna().sum())
# print("\n")
# print("*****In the test set*****")
# print(test.isna().sum())


# Fill missing values with mean column values in the train set
train.fillna(train.mean(numeric_only = True), inplace=True)
# Fill missing values with mean column values in the test set
test.fillna(test.mean(numeric_only = True), inplace=True)
# Print out changes

print("\nAfter filling in missing values")
print("\n*****In the train set*****")
print(train.isna().sum())
print("\n")
print("*****In the test set*****")
print(test.isna().sum())

# Sample Values of mixed numeriv and alphanumeric data types (tickets etc..)
print("\n Sample values mixed num/alp: ")
print(train['Ticket'].head())
print("\n")
print(train['Cabin'].head())

# Survival count of passenger with the following features:
# P-class = Ticket class:
print("\n")
print(train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False))
# Sex:
print("\n")
print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# Sibsp = number of siblings/spouses aboard:
print("\n")
print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# Nr of parents/children (parch):
print("\n")
print(train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# Plotting Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()
# Plotting Pclass and Survived relation
grid = sns.FacetGrid(train, col='Survived', row='Pclass', aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

# Feed all numerical data to model (K-means-model)
print("\n")
print(train.info())

# Drop unnecessary features that doesn't impact survival-rate
print("\nDropped unnecessary features!")
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

# Converting Sex-feature into numeric
labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])
# Write out remaing data (test does not have Survived)
print(train.info())
print("\n")
test.info()

# ********TRAIN K-MEANS MODEL********

# Drop Survival column from data
X = np.array(train.drop(['Survived'], axis=1, inplace=False).astype(float))
y = np.array(train['Survived'])
print("\n Survival is gone!!!")
train.drop(columns=['Survived'], inplace=True)  # Actually remove Survived (not in instruction)
print(train.info())

# The two clusters, survived and dead
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0)

# Percentage of passenger records that were clusterad correclty 
correct = 0
for i in range(len(X)):
 predict_me = np.array(X[i].astype(float))
 predict_me = predict_me.reshape(-1, len(predict_me))
 prediction = kmeans.predict(predict_me)
 if prediction[0] == y[i]:
    correct += 1
print(correct/len(X))

# Tweaking values to get better results using SciKit-learn implementation
kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'lloyd')
kmeans.fit(X)
KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=600,
 n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0)
correct = 0
for i in range(len(X)):
 predict_me = np.array(X[i].astype(float))
 predict_me = predict_me.reshape(-1, len(predict_me))
 prediction = kmeans.predict(predict_me)
 if prediction[0] == y[i]:
    correct += 1
print(correct/len(X))

# Tweaking even further by scalig the values correctly
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)
KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=600, n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0)
correct = 0
for i in range(len(X)):
 predict_me = np.array(X[i].astype(float))
 predict_me = predict_me.reshape(-1, len(predict_me))
 prediction = kmeans.predict(predict_me)
 if prediction[0] == y[i]:
    correct += 1
print(correct/len(X))




