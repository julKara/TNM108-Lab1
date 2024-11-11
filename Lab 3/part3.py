import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE

# Load dataset
boston = pd.read_csv('Lab 3\Boston.csv')
raw_df = pd.read_csv('Lab 3\Boston.csv', sep="\s+", skiprows=22, header=None)
data = np.hstack([boston.values[::2, :], boston.values[1::2, :2]])
target = boston.values[1::2, 2]

#from sklearn.datasets import load_boston
#boston = load_boston()
X=data
Y=target
cv = 10

#Create the different models

print('\nlinear regression')
lin = LinearRegression()
scores = cross_val_score(lin, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lin, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nridge regression')
ridge = Ridge(alpha=1.0)
scores = cross_val_score(ridge, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(ridge, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nlasso regression')
lasso = Lasso(alpha=0.1)
scores = cross_val_score(lasso, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lasso, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\ndecision tree regression')
tree = DecisionTreeRegressor(random_state=0)
scores = cross_val_score(tree, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(tree, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nrandom forest regression')
forest = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(forest, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(forest, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nlinear support vector machine')
svm_lin = svm.SVR(epsilon=0.2,kernel='linear',C=1)
scores = cross_val_score(svm_lin, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(svm_lin, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nsupport vector machine rbf')
clf = svm.SVR(epsilon=0.2,kernel='rbf',C=1.)
scores = cross_val_score(clf, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(clf, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nknn')
knn = KNeighborsRegressor()
scores = cross_val_score(knn, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(knn, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))



# prints the 4 best features selected by RFE for linear regression 
best_features=4
rfe_lin = RFE(lin,n_features_to_select=best_features).fit(X,Y)
supported_features=rfe_lin.get_support(indices=True)
for i in range(0, 4):
 z=supported_features[i]
 print(i+1,boston.columns[z])
 
print('feature selection on linear regression')
rfe_lin = RFE(lin,n_features_to_select=best_features).fit(X,Y)
mask = np.array(rfe_lin.support_)
scores = cross_val_score(lin, X[:,mask], Y, cv=cv)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lin, X[:,mask],Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('feature selection ridge regression')
rfe_ridge = RFE(ridge,n_features_to_select=best_features).fit(X,Y)
mask = np.array(rfe_ridge.support_)
scores = cross_val_score(ridge, X[:,mask], Y, cv=cv)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(ridge, X[:,mask],Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('feature selection on lasso regression')
rfe_lasso = RFE(lasso,n_features_to_select=best_features).fit(X,Y)
mask = np.array(rfe_lasso.support_)
scores = cross_val_score(lasso, X[:,mask], Y, cv=cv)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lasso, X[:,mask],Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('feature selection on decision tree')
rfe_tree = RFE(tree,n_features_to_select=best_features).fit(X,Y)
mask = np.array(rfe_tree.support_)
scores = cross_val_score(tree, X[:,mask], Y, cv=cv)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(tree, X[:,mask],Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('feature selection on random forest')
rfe_forest = RFE(forest,n_features_to_select=best_features).fit(X,Y)
mask = np.array(rfe_forest.support_)
scores = cross_val_score(forest, X[:,mask], Y, cv=cv)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(forest, X[:,mask],Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('feature selection on linear support vector machine')
rfe_svm = RFE(svm_lin,n_features_to_select=best_features).fit(X,Y)
mask = np.array(rfe_svm.support_)
scores = cross_val_score(svm_lin, X[:,mask], Y, cv=cv)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(svm_lin, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))



import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

df = pd.read_csv('Lab 3\data_cars.csv', header=None)

for i in range(len(df.columns)):
 df[i] = df[i].astype('category')
df.head()

#map catgories to values
map0 = dict( zip( df[0].cat.categories, range( len(df[0].cat.categories ))))
map1 = dict( zip( df[1].cat.categories, range( len(df[1].cat.categories ))))
map2 = dict( zip( df[2].cat.categories, range( len(df[2].cat.categories ))))
map3 = dict( zip( df[3].cat.categories, range( len(df[3].cat.categories ))))
map4 = dict( zip( df[4].cat.categories, range( len(df[4].cat.categories ))))
map5 = dict( zip( df[5].cat.categories, range( len(df[5].cat.categories ))))
map6 = dict( zip( df[6].cat.categories, range( len(df[6].cat.categories ))))
cat_cols = df.select_dtypes(['category']).columns
df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)
df = df.iloc[np.random.permutation(len(df))]
print(df.head())

df_f1 = pd.DataFrame(columns=['method']+sorted(map6, key=map6.get))
df_precision = pd.DataFrame(columns=['method']+sorted(map6, key=map6.get))
df_recall = pd.DataFrame(columns=['method']+sorted(map6, key=map6.get))

def CalcMeasures(method,y_pred,y_true,df_f1=df_f1,df_precision=df_precision,df_recall=df_recall):
 df_f1.loc[len(df_f1)] = [method]+list(f1_score(y_true,y_pred,average=None, zero_division=np.nan))
 df_precision.loc[len(df_precision)] = [method]+list(precision_score(y_pred,y_true,average=None))
 df_recall.loc[len(df_recall)] = [method]+list(recall_score(y_pred,y_true,average=None, zero_division=np.nan))
X = df[df.columns[:-1]].values
Y = df[df.columns[-1]].values

cv = 10
method = 'linear support vector machine'
clf = svm.SVC(kernel='linear',C=50)
y_pred = cross_val_predict(clf, X,Y, cv=cv)
CalcMeasures(method,y_pred,Y)
method = 'naive bayes'
clf = MultinomialNB()
y_pred = cross_val_predict(clf, X,Y, cv=cv)
CalcMeasures(method,y_pred,Y)
method = 'logistic regression'
clf = LogisticRegression()
y_pred = cross_val_predict(clf, X,Y, cv=cv)
CalcMeasures(method,y_pred,Y)
method = 'k nearest neighbours'
clf = KNeighborsClassifier(weights='distance',n_neighbors=5)
y_pred = cross_val_predict(clf, X,Y, cv=cv)
CalcMeasures(method,y_pred,Y)

print(df_f1)

print(df_precision)

print(df_recall)

labels_counts=df[6].value_counts()
print(pd.Series(map6).map(labels_counts))