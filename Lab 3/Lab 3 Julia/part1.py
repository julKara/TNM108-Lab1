import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Setting up data
from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='RdBu')
plt.show()


# Create Gaussioan distrubution with no covariance
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)

# New data and predict label
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

# Plot new data a get idea of decision boundrary
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:,0], Xnew[:,1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)
plt.show()  # Can see a slight curve that sepperates

# Make array where the columns gives the posterior probabilities of first and second label
yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)

#######Unfolding Gaussian Naive Bayes Classification#######
# We are going to make a model that predict flower species based on petal and sepal features
# We have 4 dimensional features that define 3 flower species
# More info on page 9

# 1. Preparing data
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import statistics
from math import pi
from math import e

# Load Iris dataset and spilt into train- and test-set
iris = datasets.load_iris()
data=iris.data
target=iris.target
target_values=np.unique(target)
X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.3)

# Group data by placing all individuall instances of each class in the same group
def group_by_class(self, data, target):
    """
    :param data: Training set
    :param target: the list of class labels labelling data
    :return:
    Separate the data by their target class; that is, create one group
    for every value of the target class. It returns all the groups
    """
    separated = [[x for x, t in zip(data, target) if t == c]
        for c in self.target_values]
    groups=[np.array(separated[0]),np.array(separated[1]),
        np.array(separated[2])]
    return np.array(groups)

# 2. Build model & 3. Test model
# 2. Check page 11 for what features and classes we have and page 12 for formula
# 3. Testing the model and predicting a class given the new data.

# Prior probability - probability of each class occuring
"""
The probability of each group of instances (that is the class) with
respect to the total number of instances
"""
#len(group)/len(data)

# Gauss-Naive Bayes
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import statistics
from math import pi
from math import e

class GaussNB:
    summaries={}
    target_values=[]
    
    def __init__(self):
        pass

    def group_by_class(self, data, target):
        """
        :param data: Training set
        :param target: the list of class labels labelling data
        :return:
        Separate the data by their target class; that is, create one group for every value of the target class. It returns all the groups        
        """
        separated = [[x for x, t in zip(data, target) if t == c] for c in self.target_values]
        groups=[np.array(separated[0]),np.array(separated[1]),np.array(separated[2])]
        return np.array(groups)

    def summarize(self,data):
        """
        :param data: a dataset whose rows are arrays of features
        :return:
        the mean and the stdev for each feature of data.
        """
        for index in range(data.shape[1]):
            feature_column=data.T[index]
            yield{'stdev': statistics.stdev(feature_column),'mean': statistics.mean(feature_column)}

    # Prior probability (Page 13) - learn from train set, calculate mean and standard deviation
    def train(self, data, target):
        """
        :param data: a dataset
        :param target: the list of class labels labelling data
        :return:
        For each target class:
            1. yield prior_prob: the probability of each class. P(class) eg P(Iris-virginica)
            2. yield summary: list of {'mean': 0.0,'stdev': 0.0} for every feature in data
        """
        groups = self.group_by_class(data, target)
        for index in range(groups.shape[0]):
            group=groups[index]
            self.summaries[self.target_values[index]] = {
                'prior_prob': len(group)/len(data),
                'summary': [i for i in self.summarize(group)]
            }

    # Likelihood (Page 13) - take product of all Normal Probabilities
    def normal_pdf(self, x, mean, stdev):
        """
        :param x: the value of a feature F
        :param mean: Âµ - average of F
        :param stdev: Ïƒ - standard deviation of F
        :return: Gaussian (Normal) Density function.
        N(x; Âµ, Ïƒ) = (1 / 2Ï€Ïƒ) * (e ^ (xâ€“Âµ)^2/-2Ïƒ^2
        """
        variance = stdev ** 2
        exp_squared_diff = (x - mean) ** 2
        exp_power = -exp_squared_diff / (2 * variance)
        exponent = e ** exp_power
        denominator = ((2 * pi) ** .5) * stdev
        normal_prob = exponent / denominator
        return normal_prob

    # Marginal probability (Page 16) - sum of all joint probabilities, "do not have to use it", output p16
    def marginal_pdf(self, joint_probabilities):
        """
        :param joint_probabilities: list of joint probabilities for each feature
        :return:
        Marginal Probability Density Function (Predictor Prior Probability)
        Joint Probability = prior * likelihood
        Marginal Probability is the sum of all joint probabilities for all classes.
        marginal_pdf =
          [P(setosa) * P(sepal length | setosa) * P(sepal width | setosa) * P(petal length | setosa) * P(petal width | setosa)]
        + [P(versicolour) * P(sepal length | versicolour) * P(sepal width | versicolour) * P(petal length | versicolour) * P(petal width | versicolour)]
        + [P(virginica) * P(sepal length | verginica) * P(sepal width | verginica) * P(petal length | verginica) * P(petal width | verginica)]
        """
        marginal_prob = sum(joint_probabilities.values())
        return marginal_prob

    # Joint probability (Page 14) - product of Prior Probability and Likelihood, output on p15
    def joint_probabilities(self, data):
        """
        :param data: dataset in a matrix form (rows x col)
        :return:
        Use the normal_pdf(self, x, mean, stdev) to calculate the Normal Probability for each feature
        Yields the product of all Normal Probabilities and the Prior Probability of the class.
        """
        joint_probs = {}
        for y in range(self.target_values.shape[0]):
            target_v=self.target_values[y]
            item=self.summaries[target_v]
            total_features = len(item['summary'])
            likelihood = 1
            for index in range(total_features):
                feature = data[index]
                mean = self.summaries[target_v]['summary'][index]['mean']
                stdev = self.summaries[target_v]['summary'][index]['stdev']**2
                normal_prob = self.normal_pdf(feature,mean,stdev)   # Use the Normal Distribution to calculate the Normal Probability of each feature; N(x; µ, σ)
                likelihood *= normal_prob
            prior_prob = self.summaries[target_v]['prior_prob']
            joint_probs[target_v] = prior_prob * likelihood # Take the product of the Prior Probability and the Likelihood.
        return joint_probs

    # Posterior probability (Page 17) - the result from formula
    def posterior_probabilities(self, test_row):
        """
        :param test_row: single list of features to test; new data
        :return:
        For each feature (x) in the test_row:
            1. Calculate Predictor Prior Probability using the Normal PDF N(x; Âµ, Ïƒ). eg = P(feature | class)
            2. Calculate Likelihood by getting the product of the prior and the Normal PDFs
            3. Multiply Likelihood by the prior to calculate the Joint Probability.
        E.g.
        prior_prob: P(setosa)
        likelihood: P(sepal length | setosa) * P(sepal width | setosa) * P(petal length | setosa) * P(petal width | setosa)
        joint_prob: prior_prob * likelihood
        marginal_prob: predictor prior probability
        posterior_prob = joint_prob/ marginal_prob
        Yields a dictionary containing the posterior probability of every class
        """
        posterior_probs = {}
        joint_probabilities = self.joint_probabilities(test_row)
        marginal_prob = self.marginal_pdf(joint_probabilities)
        for y in range(self.target_values.shape[0]):
            target_v=self.target_values[y]
            joint_prob=joint_probabilities[target_v]
            posterior_probs[target_v] = joint_prob / marginal_prob
        return posterior_probs

    # Get Maximum A Posterior (Page 18) - call the posterior_probabilities() method on a single test_row, for each test_row calculates 3 Posterior Probabilities
    def get_map(self, test_row):
        """
        :param test_row: single list of features to test; new data
        :return:
        Return the target class with the largest posterior probability
        """
        posterior_probs = self.posterior_probabilities(test_row)
        target = max(posterior_probs, key=posterior_probs.get)  # Take max out of 3 Posterior Probabilities
        return target

    # Return a prediction for each test_row.
    def predict(self, data):
        """
        :param data: test_data
        :return:
        Predict the likeliest target for each row of data.
        Return a list of predicted targets.
        """
        predicted_targets = []
        for row in data:
            predicted = self.get_map(row)
            predicted_targets.append(predicted)
        return predicted_targets

    # Test the performance by deviding correct vs wrong predictions
    def accuracy(self, ground_true, predicted):
        """
        :param ground_true: list of ground true classes of test_data
        :param predicted: list of predicted classes
        :return:
        Calculate the the average performance of the classifier.
        """
        correct = 0
        for x, y in zip(ground_true, predicted):
            if x==y:
                correct += 1
        return correct / ground_true.shape[0]

def main():
    nb = GaussNB()
    iris = datasets.load_iris()
    data=iris.data
    target=iris.target
    nb.target_values=np.unique(target)
    X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.3)
    nb.train(X_train,y_train)
    predicted = nb.predict(X_test)
    accuracy = nb.accuracy(y_test, predicted)
    print('Accuracy: %.3f' % accuracy)

main()