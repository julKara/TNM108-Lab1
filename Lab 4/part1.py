from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

# Create documents
d1 = "The sky is blue."
d2 = "The sun is bright."
d3 = "The sun in the sky is bright."
d4 = "We can see the shining sun, the bright sun."
Z = (d1,d2,d3,d4)

print(vectorizer)

# Define stop words (words that are ignored)
my_stop_words={"the","is"}

# Define vocabulary (the words we care about)
my_vocabulary={'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}

# Add vocabulary and stop words to the vectorizer
vectorizer=CountVectorizer(stop_words=my_stop_words,vocabulary=my_vocabulary)

print(vectorizer.vocabulary)
print(vectorizer.stop_words)

# Create a sparse matrix
smatrix = vectorizer.transform(Z)
print(smatrix)

# Convert to dense matrix
matrix = smatrix.todense()
print(matrix)


# ----------------------------- tf-idf ---------------------------------

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(norm="l2")
tfidf_transformer.fit(smatrix)

# print idf values
feature_names = vectorizer.get_feature_names_out()
import pandas as pd
df_idf=pd.DataFrame(tfidf_transformer.idf_, index=feature_names,columns=["idf_weights"])
# sort ascending
df_idf.sort_values(by=['idf_weights'])

print(df_idf)

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(smatrix)


# get tfidf vector for first document
first_document = tf_idf_vector[0] # first document "The sky is blue."
# print the scores
df=pd.DataFrame(first_document.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
print(df) # The lower the score the more common the word is (0 means it does'nt appear)



# --------------------------------- Document similarity -------------------------------------

d1 = "The sky is blue."
d2 = "The sun is bright."
d3 = "The sun in the sky is bright."
d4 = "We can see the shining sun, the bright sun."
Z = (d1,d2,d3,d4)

# Fit documents into matrix
import math
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
print(tfidf_matrix.shape)

# Calculate cosine similarity for the first document
from sklearn.metrics.pairwise import cosine_similarity
cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
print(cos_similarity) # Higer value means more similar (1 is identical)

# Take the cos similarity of the third document (cos similarity=0.52)
angle_in_radians = math.acos(cos_similarity[0,2]) # changed to [0,2] since its a  matrix, row 0 column 2
print(math.degrees(angle_in_radians))


# --------------------------- Classifying text --------------------------------------

from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names

# Define categories and test and train set
my_categories = ['rec.sport.baseball','rec.motorcycles','sci.space','comp.graphics']
train = fetch_20newsgroups(subset='train', categories=my_categories)
test = fetch_20newsgroups(subset='test', categories=my_categories)

print(len(train.data))
print(len(test.data))
print(train.data[9])

# Create sparse matrix from train data
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_counts=cv.fit_transform(train.data)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)

# Create multinomial naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train_tfidf, train.target)

# Apply model
docs_new = ['Pierangelo is a really good baseball player','Maria rides her motorcycle', 'OpenGL on the GPU is fast', 'Pierangelo rides his motorcycle and goes to play football since he is a good football player too.']
X_new_counts = cv.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = model.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
 print('%r => %s' % (doc, train.target_names[category]))