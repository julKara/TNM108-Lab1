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
                normal_prob = self.normal_pdf(feature,mean,stdev)
                likelihood *= normal_prob
            prior_prob = self.summaries[target_v]['prior_prob']
            joint_probs[target_v] = prior_prob * likelihood
        return joint_probs

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

    def get_map(self, test_row):
        """
        :param test_row: single list of features to test; new data
        :return:
        Return the target class with the largest posterior probability
        """
        posterior_probs = self.posterior_probabilities(test_row)
        target = max(posterior_probs, key=posterior_probs.get)
        return target

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

if __name__ == '__main__':
    main()