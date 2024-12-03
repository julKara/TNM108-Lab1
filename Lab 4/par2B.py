from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']

# Make test and train set
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)

######## Building the pipeline ########
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Arbitrary names, used to perform grid-searches
text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', MultinomialNB()),
])

# Train model with single command
text_clf.fit(twenty_train.data, twenty_train.target)


######## Evaluating performance of test set ########
import numpy as np

docs_test = twenty_test.data

predicted = text_clf.predict(docs_test)

# Gives 83.5% accurancy
print("multinomialBC accuracy ",np.mean(predicted == twenty_test.target))

# training SVM classifier
from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
,max_iter=5, tol=None)),
])

text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(docs_test)

# Gives 91.3% accuracy
print("SVM accuracy ",np.mean(predicted == twenty_test.target))

# More detailed performance stats
from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted, 
                                    target_names=twenty_test.target_names))

# Previous but as confusion matrix
print(metrics.confusion_matrix(twenty_test.target, predicted))

######## Parameter tuning using grid search ########
from sklearn.model_selection import GridSearchCV

parameters = {
 'vect__ngram_range': [(1, 1), (1, 2)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (1e-2, 1e-3),    # Use either alfa 0.01 or 0.001 (penalty parameter)
}

# Try to use 8 cores and run in parallell
gs_clf = GridSearchCV(text_clf, parameters, cv = 5, refit = True, n_jobs = -1) # if parameter value is -1, detect how many cores

# Perfomrm search on smaller subset of training data
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

# Result calling on GridSearchCV classifer
print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])

# Result of mean of objects best score and params-attriutes
print(gs_clf.best_score_)   # Gives 91%

for param_name in sorted(parameters.keys()):
 print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
