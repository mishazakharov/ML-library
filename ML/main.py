import numpy as np
from random import randrange
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn
from metrics import accuracy_score
import random as rd
from math import sqrt
import cvxopt
from kernelz import *
import math



def check_data_format(X):
    ''' Check a format of data of input algorithm '''
    nparray = lambda x: isinstance(x,np.ndarray)
    if not nparray(X):
        X = X.values

    return X



class LinearRegression(object):
    """ Class that creates a basic non-regularized linear regression model.

    Args:
        learning_rate(int): it is a constant used gradient descent
        number_of_iterations(int): is number of iterations in the process of 
            teaching our model, which is just all about adjusting weights of
            the model
        normalize(bool): flag for normalization

    """
    def __init__(self,learning_rate=0.00001,number_of_iterations=3000,
                                                normalize=False):
        # Initialization of variables
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.theta = None
        self.normalize = normalize

    def step_gradient(self,old_theta,vector_x,vector_y):
        # Number of samples
        m = len(vector_y)
        # Vector full of parameters
        theta = old_theta
        # Vector delta(appears from vectorization)
        delta = np.zeros((vector_x.shape[0],1))
        for i in range(0,m):
            # Define vectors x and y 
            x_i = vector_x
            y_i = vector_y
            x_i = x_i[:,i].reshape(vector_x.shape[0],1)
            y_i = y_i[i]
            # Hypothesis
            hypothesis = np.dot(theta.T,x_i)
            constant = (1/m) * (float(hypothesis) - y_i)
            delta += (x_i) * constant
            # Updating weights
            new_theta = old_theta - (self.learning_rate * delta)

        return new_theta

    def gradient_runner(self,initial_theta,vector_x,vector_y):
        # function to train our model
        theta = initial_theta
        for i in range(self.number_of_iterations):
            theta = self.step_gradient(theta,vector_x,vector_y)

        return theta

    def fit(self,X,y):
        X,y = check_data_format(X),check_data_format(y)
        # Normalizing data by subtracting mean and debiding by l2 norm
        if self.normalize:
            X = (X - np.mean(X)) / np.linalg.norm(X)
        # n+1 x m dim matrix
        x_i = X.T
        # Adding bias
        vector_x = np.insert(x_i,0,[1],axis=0)
        vector_y = y
        # "Random" weights initialization!
        initial_theta = np.zeros((vector_x.shape[0],1))
        self.theta = self.gradient_runner(initial_theta,vector_x,vector_y)

    def predict(self,x):
        x = check_data_format(x)
        # Normalizing data by subtracting mean and deviding by l2 norm
        if self.normalize:
            x = (x - np.mean(x)) / np.linalg.norm(x)
        # List for our predictions
        predictions = []
        # Defining size
        m = x.shape[0]
        # Predicting on each sample, adding it to our container-list
        for i in range(0,m):
            x_i = x[i]
            # Adding bias
            x_i = np.insert(x_i,0,[1],axis=0)
            # Prediction
            prediction = np.dot(self.theta.T,x_i)
            predictions.append(prediction)
        # Answer
        predictions = np.array(predictions).reshape(-1,1)

        return predictions

    def get_weights(self):
        # Method that returns weights of our model
        return self.theta



class LogisticRegression(object):
    """ This class creates basic non-regularized logistic regression model.

    Args:
        learning_rate(int): used in gradient descent
        number_of_iterations(int): is number of iterations in the process of 
          teaching our model, which is just all about adjusting weights of
          the model

    """
    def __init__(self,learning_rate=0.00159,number_of_iterations=3000):
        
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.theta = None

    def step_gradient(self,old_theta,vektor_x,vektor_y):
        m = len(vektor_y)
        # Weights
        theta = old_theta
        # Delta
        delta = np.zeros((vektor_x.shape[0],1))
        for i in range(0,m):
            # Creating n+1 dim vector(adding bias)
            x_i = vektor_x
            y = vektor_y
            # N-dim vector which consist of all attributes of 1 sample
            x_i = x_i[:,i].reshape(vektor_x.shape[0],1)
            # Label on this sample
            y_i = y[i]
            # Argument of our sigmoid function
            z = np.dot(theta.T,x_i)
            # Sigmoid function
            hypothesis = 1/(1 + np.exp(-z))
            # Constants
            constant = (1/m) * (float(hypothesis) - y_i)
            # New delta
            delta += (x_i) * constant
        # Gradient descent
        new_theta = old_theta - (self.learning_rate * delta)
        return new_theta
        # When we call this method old_theta = new_theta of the 
        # past step(right way of updationg our weights!

    def gradient_runner(self,initial_theta,vektor_x,vektor_y):
        theta = initial_theta
        for i in range(self.number_of_iterations):
            theta = self.step_gradient(theta,vektor_x,vektor_y)
        return theta

    def fit(self,X,y):
        X,y = check_data_format(X),check_data_format(y)
        x_i = X.T
        vektor_x = np.insert(x_i,0,[1],axis=0)
        vektor_y = y
        initial_theta = np.zeros((vektor_x.shape[0],1))
        self.theta = self.gradient_runner(initial_theta,vektor_x,vektor_y)

    def predict(self,x):
        x = check_data_format(x)
        m = x.shape[0]
        predictions = list()
        for i in range(0,m):
            # Prediction vector
            x_i = x[i]
            vektor_x = np.insert(x_i,0,[1],axis=0)
            z = np.dot(self.theta.T,vektor_x)
            prediction = 1/(1 + np.exp(-z))
            # Threshold
            if prediction > 0.5:
                prediction = 1
            else:
                prediction = 0
            predictions.append(prediction)
        predictions = np.array(predictions).reshape(-1,1)
        return predictions



class SVM(object):
    ''' Support Vector Machine '''
    def __init__(self):
        raise NotImplementedError



# Class for decision tree!
class Question(object):
    ''' Question is used for data partition 
    Class track number of
    Method "match" asks question and returns True if answer is "yes".
    '''
    def __init__(self,column,value):
        self.column = column
        self.value = value

    def match(self,example):
        # Asks a question
        b = DecisionTree()
        val = example[self.column]
        if b.is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This special method helps to print everythin out in
        # a readbl way
        condition = '=='
        if is_numeric(self.value):
            condition = '>='
        return 'Is %s %s %s?' % (
                header[self.column],condition,str(self.value))

# Leaf for decision tree
class Leaf(object):
    ''' Leaf node classifies data.
    It contain a dictionary(key - class,value - number of samples of this class)
    '''
    def __init__(self,rows):
        b = DecisionTree()
        self.predictions = b.class_counts(rows)


class Decision_Node(object):
    ''' Decision Node asks a question
    Contain reference to a question and 2 nodes
    '''
    def __init__(self,question,true_branch,false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree(object):
    """ This class creates simple decision tree

    Args:
        X(np.ndarray/pd.DataFrame): MxN training matrix
        y(np.ndarray/pd.DataFrame): M training vecotr of labels/values
        
    """ 
    def __init__(self):
        pass

    def unique_vals(self,rows,col):
        ''' Returns number of unique elemts of a column '''
        return set([row[col] for row in rows])

    def class_counts(self,rows):
        '''
        Counts number of samples of a class in learning data
        Returns a dictionary, where key - class, value - number
        '''
        counts = {}
        for row in rows:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
                counts[label] += 1
        return counts

    def is_numeric(self,value):
        ''' Returns True if input data is number, else False '''
        return isinstance(value,int) or isinstance(value,float)

    def partition(self,rows,question):
        ''' Partition of our dataset
        For every row in data
        '''
        true_rows,false_rows = [],[]
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows,false_rows

    def gini(self,rows):
        ''' Counts gini index '''
        counts = self.class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl ** 2
        return impurity

    def information_gain(self,left,right,current_uncertainty):
        ''' Counts information gain
        Uncertainty of the first node minus weighted uncertainties 
        of two next nodes(DOCHERNIX)
        '''
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - (p * self.gini(left) - 
                                    (1 - p) * self.gini(right))

    def find_best_split(self,rows):
        ''' Finds best split with a brute-force method(checking every attri-
        bute, counting information gain for every partition etc.
        '''
        # Contains best value of inf_gain
        best_gain = 0
        # Contains best question
        best_question = None
        # Uncertainty of first node
        current_uncertainty = self.gini(rows)
        # Number of attributes
        n_features = len(rows[0]) - 1
        # For every attribute
        for col in range(n_features):
            # Contatins unique values
            values = set([row[col] for row in rows])
            # For every value of attribute
            for val in values:
                question = Question(col,val)
                # Partition of data based on a current question
                true_rows,false_rows = self.partition(rows,question)
                # If data is not partitianing with this question
                # we just skip that value
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                # Counting inf_gain after partition
                gain = self.information_gain(true_rows,false_rows,
                                                current_uncertainty)
                # Updationg the best gain and the best question
                if gain >= best_gain:
                    best_gain,best_question = gain,question
        return best_gain,best_question

    def fit(self,X,y):
        ''' Builds a tree '''
        # no API
        rows = np.c_[X,y]
        # Finding best question and best gain
        gain,question = self.find_best_split(rows)
        # If gain = 0 then we cant ask question anymore
        # that's why we return leaf.(Base of a recursive function)
        if gain == 0:
            return Leaf(rows)
        # If we are here then we ve already found attribue/value
        # for partition
        true_rows,false_rows = self.partition(rows,question)
        true_X = np.delete(true_rows,-1,1)
        true_y = np.array(true_rows)[:,-1]
        false_X = np.delete(false_rows,-1,1)
        false_y = np.array(false_rows)[:,-1]
        # Recursively build true branch and false branch
        true_branch = self.fit(true_X,true_y)
        false_branch = self.fit(false_X,false_y)
        # Return question node
        # Keeps track of the best attribute/value and what branches to follow
        return Decision_Node(question,true_branch,false_branch)

    def classify(self,row,node):
        # Base
        if isinstance(node,Leaf):
            return node.predictions
        if node.question.match(row):
            return self.classify(row,node.true_branch)
        else:
            return self.classify(row,node.false_branch)

    def predict(self,test_data,my_tree):
        exes = []
        for row in test_data:
            b = (self.classify(row,my_tree))
            for x in b.keys():
                exes.append(x)

        return np.array(exes).reshape(-1,1)



class RandomForest(object):
    """ Class creates simple random forest using basic bagging.

    Bagging is used to get dissimilar models, only ensemble of dissimilar
    models can perform better than a single model. You can use either bagging
    or boosting to create random forests. RF is basically just an ensemble of 
    decision trees.

    Args:
        n_estimators(int): number of decision trees in the forest.
        criterion(str): criterion used in estmation of data partition 
        max_depth(int): maximal depth for partition

    """
    def __init__(self,n_estimators=3,criterion='gini',max_depth=None):
        ''' Initializing variables! '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.predictions = []
        self.trees = []

    def subsample(self,dataset_x,dataset_y,ratio):
        ''' Function that 
        '''
        dataset = np.c_[dataset_x,dataset_y]
        sample = list()
        n_sample = round(len(dataset) * ratio)
        while len(sample) < n_sample:
            index = randrange(len(dataset))
            sample.append(dataset[index])
        return sample

    def fit(self,X,y):
        ''' Building random forest '''
        X,y = check_data_format(X),check_data_format(y)
        for i in range(self.n_estimators):
            # Preparing data (simple bagging)
            new_data = np.array(self.subsample(X,y,0.5))
            # Extracting X and y from new_data
            new_y = new_data[:,-1].reshape(-1,1)
            new_X = np.delete(new_data,-1,1)
            tree = DecisionTree()
            # Worst realization!
            b = tree.fit(new_X,new_y)
            self.trees.append(b)

    def predict(self,predict_on_X):
        ''' Prediction of our RF!
        '''
        predict_on_X = check_data_format(predict_on_X)
        conclusion = np.zeros((predict_on_X.shape[0],1))
        tree = DecisionTree()
        for b in self.trees:
            prediction = tree.predict(predict_on_X,b)
            conclusion = conclusion + prediction

        conclusion = conclusion / int(len(self.trees))
        # Return rounded vector
        return np.round(conclusion)



class Bagging(object):
    ''' Ensemble method - bagging. '''
    def __init__(self):
        raise NotImplementedError


            
class AdaBoostClassifier(object):
    """ This class realizes esmeble method called ada boost.

    This method consistently trains classifiers, with each successive 
    classifier paying more attention to incorrectly related samples.
    This one works only with 1/-1 labels and created with sklearn.tree.
    .DecisionTree 

    Args:
        n_estimators(int): number of classifiers in ensemble
        base_estimator(class): base type of classifier used in ensemble

    """
    def __init__(self,n_estimators=11,lr=0.5,base_estimator=''):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.lr = lr

    def prepare_data(self,y):
        """ This method is used to prepare data i.e. turn 0 to -1
        """
        y = np.where(y == 0,-1,1)

        return y

    def fit(self,X,Y):
        X,Y = check_data_format(X),check_data_format(Y)
        # Turning all 0s into -1s
        Y = self.prepare_data(Y)
        # List for classifiers
        self.models = []
        # Alpha - weight of every classifier
        self.alphas = []
        N = len(X)
        # Weights of each sample(1/N by default)
        W = np.ones(N)/N
        for m in range(self.n_estimators):
            if self.base_estimator == '':
                # Default classifier - DT
                tree = DecisionTreeClassifier(max_depth=1)
                tree.fit(X,Y,sample_weight=W)
                P = tree.predict(X)
            # Error is a sum of missclassified samples               
            err = W.dot(P != Y)
            alpha = self.lr * (math.log(1 - err) - math.log(err + 1e-10)) 
            # Vectorized form
            #W = W * np.exp(-alpha*Y*P)
            # I am not sure if I am updating the weights right!
            # But the running version is much better than commented one!!!!
            W *= np.exp(alpha * Y * ((W > 0) | (alpha < 0)))
            # Normalizing
            W = W / W.sum()
            self.models.append(tree)
            self.alphas.append(alpha)

    def predict(self,X):
        ''' Class for classifing '''
        X = check_data_format(X)
        N = len(X)
        FX = np.zeros(N)
        for alpha,tree in zip(self.alphas,self.models):
            FX += alpha * tree.predict(X)
        # The way to get prediction is pretty ridiculous
        average = np.average(FX)
        FX = np.where(FX < average,0,1)

        return FX



class StackingRegression(object):
    """ Class realizes ensemble method - stacking regression

    Meta learner - linear regression. Meta learner is used to agrigate
    predictions of models in previous layer.

    Args:
        list_of_models(list): list contains models that we want to use
            in the first layer, base models.

    """
    def __init__(self,list_of_models,meta_learner=''):
        # List of ensemble models
        self.list_of_models = list_of_models
        # List to keep track of predictions on validation dataset
        self.predictions_on_valid = []
        # List to keep track of predictions on test dataset
        self.predictions_on_test = []
        # Meta learner
        self.meta_learner = meta_learner
        if self.meta_learner == '':
            self.meta_learner = sklearn.linear_model.LinearRegression()

    def fit(self,X,y):
        X,y = check_data_format(X),check_data_format(y)
        train_X,valid_X,train_y,valid_y = train_test_split(X,y,test_size=0.5)
        for model in self.list_of_models:
            # Predictions on validation dataset
            model.fit(train_X,train_y)
            prediction = model.predict(valid_X)
            self.predictions_on_valid.append(prediction)

        # Making it a global variable for the whole class!
        #self.valid_y = valid_y
        #self.train_X = train_X
        #self.train_y = train_y

        # Transforming in np.array
        new = np.zeros((self.predictions_on_valid[0].shape[0],1))
        for prediction in self.predictions_on_valid:
            new = np.c_[new,prediction]
        # Deleting zero column
        new_X = np.delete(new,0,1)
        self.meta_learner.fit(new_X,valid_y.reshape(-1,1))

    def predict(self,X):
        ''' If we use sklearn.linear_model.LinearRegression
        then MSE is much lower !!! 
        '''
        X = check_data_format(X)
        # Predictions on test dataset
        for model in self.list_of_models:
            #model.fit(self.train_X,self.train_y)
            prediction_on_test = model.predict(X)
            self.predictions_on_test.append(prediction_on_test)

        # Transforming our predictions in np.array
        new_test = np.zeros((self.predictions_on_test[0].shape[0],1))
        for prediction in self.predictions_on_test:
            new_test = np.c_[new_test,prediction]
        # Deleting 0 column
        X_test = np.delete(new_test,0,1)
        predict_on_test = self.meta_learner.predict(X_test)

        return prediction_on_test



class VotingClassifier(object):
    """ Class creates basic voting classifier

    Basic voting classifer just creates bigger model, this one doesn't 
    really care about any dissimilarity, just makes a bigger model
    This one actually does "hard" classification!

    Args:
        estimators(list): list contains base classifiers that we want to use

    """ 
    def __init__(self,estimators):
        # List of models
        self.estimators = estimators
        # Keeping fit's of all our models
        self.models = []

    def fit(self,X,y):
        for each_model in self.estimators:
            fit = each_model.fit(X,y)
            self.models.append(fit)

    def predict(self,X):
        #
        predictions = np.zeros((X.shape[0],1))
        for number,model in enumerate(self.estimators):
            try:
                prediction = model.predict(X).reshape(-1,1)
            except TypeError:
                prediction = model.predict(X,self.models[number]).reshape(-1,1)
            predictions += prediction
        # Deviding by number of models
        predictions = predictions / len(self.models)
        function = lambda x: round(x)
        # Rounding values!
        predictions = np.round(predictions)

        return predictions



class BaggingClassifier(object):
    """ Class creates bagging classifier by subsampling input data.

    This ensemble method creates dissimlar classifier by subsampling input
    data, using the same base_estimator on it's every model

    Args:
        base_estimator(class): base model that we want to use
        n_estimators(int): number of base models in an ensemble

    """
    def __init__(self,base_estimator='',n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        # List with trained models
        self.list_of_trained_estimators = []

    def subsample(self,X,y,ratio):
        ''' already talked about it '''
        sample_X = list()
        sample_y = list()
        n_sample = round(len(X) * ratio)
        while len(sample_X) < n_sample:
            index = randrange(len(X))
            sample_X.append(X[index])
            sample_y.append(y[index])

        return sample_X,sample_y

    def fit(self,X,y):
        X,y = check_data_format(X),check_data_format(y)
        # Base model - decision tree
        if self.base_estimator == '':
            self.base_estimator = DecisionTreeClassifier()
        # Same principle
        ratio = 1 / self.n_estimators
        for i in range(self.n_estimators):
            new_X,new_y = self.subsample(X,y,ratio)
            b = self.base_estimator.fit(new_X,new_y)
            self.list_of_trained_estimators.append(b)

    def predict(self,X):
        X = check_data_format(X)
        predictions = np.zeros((X.shape[0],1))
        for model in self.list_of_trained_estimators:
            predictions += model.predict(X).reshape(-1,1)
        predictions = predictions / self.n_estimators
        predictions = np.array(predictions)

        return predictions



class CrossEntropy(object):
    ''' Class for cross-entropy loss function. '''
    def __init__(self):
        pass

    def mist(self,y,p):
        # Don't devide by zero
        p = np.clip(p,1e-15,1-1e-15)
        return -y*np.log(p) - (1-y)*np.log(1-p)

    def gradient(self,y,p):
        p = np.clip(p,1e-15,1-1e-15)
        p,y = p.reshape(-1,1),y.reshape(-1,1)

        return -(y/p) + (1-y) / (1 - p)
    

class MSE(object):
    ''' MSE loss function '''
    def __init__(self):
        pass

    def mist(self,y,y_pred):
        return 0.5 * (y-y_pred)**2

    def gradient(self,y,y_pred):
        return -(y-y_pred)


class GradientBoosting(object):
    """ This is a superclass for ensemble method - gradient boosting(on trees)

    Args:
        n_estimators(int): number of estimators in an ensemble
        learning_rate(int): constant representing weight of ever base model
        min_samples_split(int): min samples needed to make a partition
        min_imputiry(int): minimal number of impurity needed to make a partition
        max_depth(int): maximal depth of every base tree
        regression(bool): change basic loss function in order to use it on both
        regression and classification tasks

    """
    def __init__(self,n_estimators,learning_rate,
            min_samples_split,min_impurity,max_depth,regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.trees = []
        self.loss = MSE()
        # Initializing trees for classification
        if not self.regression:
            self.loss = CrossEntropy()
            '''
            for _ in range(self.n_estmiators):
                tree = sklearn.tree.DecisionTreeClassifier(
                        min_samples_split=self.min_samples_split,
                        min_impurity=self.min_impurity,
                        max_depth=self.max_depth)
                self.trees.append(tree)
            '''
        # Initializing trees for regression
        for _ in range(self.n_estimators):
            tree = sklearn.tree.DecisionTreeRegressor(
                    min_samples_split=self.min_samples_split,
                    min_impurity_split=self.min_impurity,
                    max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self,X,y):
        X,y = check_data_format(X),check_data_format(y)
        # Default by mean
        y_pred = np.full(np.shape(y),np.mean(y,axis=0))
        for tree in self.trees:
            # Gradient of loss function
            gradient = self.loss.gradient(y,y_pred)
            tree.fit(X,gradient)
            update = tree.predict(X)
            # Learning rate just for regularization
            y_pred -= self.learning_rate * update

    def predict(self,X):
        X = check_data_format(X)
        # The only way to form predictions in GB
        y_pred = np.array([])
        for tree in self.trees:
            update = tree.predict(X)
            update = (self.learning_rate*update)
            y_pred = -update if not y_pred.any() else y_pred - update
        # Classifiaction 
        if not self.regression:
            y_pred = np.exp(y_pred) / (1 + np.exp(y_pred))
            # rounding (threshold = 0.5)
            y_pred = np.round(y_pred)

        return y_pred



class GradientBoostingClassifier(GradientBoosting):
    ''' Gradient boosting for classification
    Just using different loss function and different way of predicting
    '''
    def __init__(self,n_estimators=200,learning_rate=0.5,min_samples_split=2,
                                min_impurity=1e-7,max_depth=4):
        super(GradientBoostingClassifier,self).__init__(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                min_samples_split=min_samples_split,
                                min_impurity=min_impurity,
                                max_depth=max_depth,
                                regression=False
                                )



class GradientBoostingRegressor(GradientBoosting):
    ''' Regression on GB '''
    def __init__(self,n_estimators=200,learning_rate=0.5,min_samples_split=2,
                                        min_impurity=1e-7,max_depth=4):
        # Enherance
        super(GradientBoostingRegressor,self).__init__(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    min_samples_split=min_samples_split,
                    min_impurity=min_impurity,
                    max_depth=max_depth,
                    regression=True)



class GridSearchCV(object):
    ''' '''
    def __init__(self,model,param):
        '''
        self.best_estimator_ = 0
        self.best_param = {}

    def fit(self,X,y):
        for key in self.param.keys():
            for value in self.param[key]:
                self.model.fit(key=value)
                prediction = self.model.predict(X)
                metrics = accuracy_score(y,prediction)
                if metric > minimal:
                    minimal = metrics
                    self.best_params[key] = value
        self.best_estimator_ = self.model(self.be=self.best+params.values()[0])
        '''
        raise NotImplementedError



class KMeans():
	def __init__(self,clusters,iterations=100):
		self.clusters = clusters
		self.iterations = iterations
		self.first = 0
		self.output = 0


	def fit(self,X):
		X = check_data_format(X)
		self.first = X
		m=X.shape[0] #number of training examples
		n=X.shape[1] #number of features. Here n=2
		n_iter=100
		K = self.clusters

		Centroids=np.array([]).reshape(n,0) 
		for i in range(K):
			rand=rd.randint(0,m-1)
			Centroids=np.c_[Centroids,X[rand]]

		for i in range(n_iter):
			#step 2.
			EuclidianDistance=np.array([]).reshape(m,0)
			for k in range(K):
				tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
				EuclidianDistance=np.c_[EuclidianDistance,tempDist]

			C=np.argmin(EuclidianDistance,axis=1)+1
		    #step 2.b
			Y = {}
			for k in range(K):
				Y[k+1]=np.array([]).reshape(2,0)
			for i in range(m):
				Y[C[i]]=np.c_[Y[C[i]],X[i]]
			for k in range(K):
				Y[k+1]=Y[k+1].T
			for k in range(K):
				Centroids[:,k]=np.mean(Y[k+1],axis=0)
		self.output = Y

	def fit_predict(self,X):
		''' Осуществляет fit, predict и выводит распределение по кластерам'''
		X = check_data_format(X)
		self.fit(X)
		return self.output

	def predict(self,X_new):
		''' Вычисляет, к какому кластеру принадлежит новая точка
		Пока что работает только для предикта на одной точке!(НАДО ИСПРАВИТЬ)

		ПОЛНОСТЬЮ НЕ РАБОТАЕТ, ВОЗВРАЩАЕТ НЕПРАВИЛЬНЫЕ КЛАССЫ!!!!!!
		РАБОТАЕТ ДАЖЕ НЕПРАВИЛЬНЫМ ОБРАЗОМ ТОЛЬКО ДЛЯ 1 НОВОЙ ТОЧКИ!!!!!
		new = np.vstack((self.first,X_new))
		self.fit(new)
		cluster_appended = 0
		for key in self.output.keys():
			for value in self.output[key]:
				if value.all() == X_new.all():
					return key
					break

		# Возвращает номер кластера, к которому была определена новая точка
		return None
		'''
		raise NotImplementedError



class KNeighborsClassifier(object):
    """ This class realizes k-neighbors classifier

    This algorithm just compares new object to k objects in train data
    by calculating euclidian distance and outputs the probability of a 
    label on a new object

    Args:
        k(int): just a number of neighbors to look at

    """
    def __init__(self,k=15):
        # k - number of neighbors to track!
        self.k = k

    def euclidian_distance(self,x1,x2):
        ''' Calculate ED between X and x '''
        summation = 0
        for i in range(len(x1)):
           summation += (x1[i] - x2[i])**2
        # Returns euclidian distance between two vectors.
        return sqrt(summation)
           

    def _vote(self,classes):
        ''' Class of voting. '''
        counts = np.bincount(classes.astype('int'))
        return counts.argmax()

    def predict(self,X_test,X_train,y_train):
        #Transforming data if needed!
        X_train = check_data_format(X_train)
        y_train = check_data_format(y_train)
        X_test = check_data_format(X_test)

        y_pred = np.empty(X_test.shape[0])
        # Determine the class
        for i,sample in enumerate(X_test):
            # sorting by euclidain_distance and showing only first self.k of them!
            neighbors = (np.argsort([self.euclidian_distance(sample,
                                        x) for x in X_train])[:self.k])
            # Extracting classes of k neighbors
            classes = np.array([y_train[i] for i in y_train])
            # ...
            y_pred[i] = self._vote(classes)

        return y_pred



class NaiveBayesClassifier():
    ''' Naive Bayes Classifier
    Info: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
    '''
    def fit(self,X,y):
        self.X = X
        self.y = y
        self.existing_classes = np.unique(y)
        self.parameters = []
        for i,c in enumerate(self.existing_classes):
            X_c = X[np.where(y == c)]
            self.parameters.append([])
            #
            for col in X_c.T:
                parameters = {"mean":col.mean(),"var":col.var()}
                self.parameters[i].append(parameters)

    def _calculate_likelihood(self,mean,var,x):
        ''' Took this from another library on github '''
        eps = 1e-4
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean,2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_prior(self,c):
        ''' Calculate apriori probability of class c '''
        prob = np.mean(self.y == c)
        return prob

    def _classify(self,sample):
        ''' Classifies using bayes theory 
        Info: https://en.wikipedia.org/wiki/Bayes%27_theorem
        '''
        posteriors = list()
        # Through list of classes:
        for i,c in enumerate(self.existing_classes):
            # We initialize this posterior probability as apriori
            posterior = self._calculate_prior(c)
            for feature_value,params in zip(sample,self.parameters[i]):
                likelihood = self._calculate_likelihood(params['mean'],
                                                params['var'],feature_value)
                posterior *= likelihood
            posteriors.append(posterior)
        # Return the class with with the largest probability value(posterior)
        return self.existing_classes[np.argmax(posteriors)]

    def predict(self,X):
        ''' Predict the class of X '''
        y_pred = [self._classify(x) for x in X]
        return y_pred

            

class MultiLayerPerceptron(object):
    ''' Creates a simple neural net with 1 hidden layers 
    Not working properly returns only 1s
    '''
    def __init__(self,x,y,neurons_in_layer=4):
        self.input = x
        self.y = y
        self.neurons_in_layer = neurons_in_layer
        # Weights of 1st layer
        self.weights1 = np.random.rand(self.input.shape[1],
                                            self.neurons_in_layer)
        # Weights of 2nd layer
        self.weights2 = np.random.rand(self.neurons_in_layer,1)
        self.output = np.zeros(y.shape)

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1.0 - x)

    def feedforward(self):
        ''' Feedforward propogation '''
        # First layer
        self.layer1 = self.sigmoid(np.dot(self.input,self.weights1))
        # Second layer
        self.output = self.sigmoid(np.dot(self.layer1,self.weights2))

    def backpropogation(self):
        ''' Backpropogation in NN '''

        # Updating weights2 
        error = self.y - self.output
        learning_rate = 2
        d_weights2 = np.dot(self.layer1.T,(learning_rate*
                            error*self.sigmoid_derivative(self.output)))
        # Updating weights1 
        error1 = np.dot(error,self.weights2.T)
        d_weights1 = np.dot(self.input.T,(learning_rate*
                                error1*self.sigmoid_derivative(self.layer1)))
        # Updating weights
        self.weights1 -= d_weights1
        self.weights2 -= d_weights2

    def fit(self,number_of_iterations):
        ''' Training our NN '''
        for i in range(number_of_iterations+1):
            self.feedforward()
            self.backpropogation()
        
    def predict(self,new_input):
        ''' Predicting new values '''
        l1 = self.sigmoid(np.dot(new_input,self.weights1))
        output = self.sigmoid(np.dot(l1,self.weights2))
        return np.round(output)



class Perceptron(object):
    ''' A single perceptron(neuron!) 
    Not working properly returns all 1s
    '''
    def __init__(self,X_train,y_train,learning_rate=0.01,
                                number_of_iterations=1000):
        np.random.seed(1)
        self.X_train = X_train
        self.y_train = y_train
        # 1 neuron with 3 inputs and 1 output
        self.synaptic_weights = 2 * np.random.random((self.X_train.shape[1],1)) - 1
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations

    def _sigmoid(self,x):
        ''' sigmoid function of x '''
        return 1/(1 + np.exp(-x))

    def _sigmoid_derivative(self,x):
        ''' derivative of sigmoid in x '''
        return x * (1-x)

    def predict(self,X_test):
        ''' predicting on X_test '''
        return np.round(self._sigmoid(np.dot(X_test,self.synaptic_weights)))

    def fit(self):
        ''' training one neuron '''
        for iteration in range(self.number_of_iterations):
            # Making a prediction on X_train
            output = self.predict(self.X_train)
            # Computing an error(the difference between output and labels)
            error = self.y_train - output
            # The adjustment
            adjustment = np.dot(self.X_train.T,(error*self._sigmoid_derivative(output)))
            # Adjusting
            self.synaptic_weights += self.learning_rate * adjustment



class SVC(object):
    ''' Support Vector Machine Classifier

    General info: https://en.wikipedia.org/wiki/Support-vector_machine
    C - Penalty
    kernel - function linear,polynomial,rbf
    power - the degree of polynomial kernel (<x1,x2> + coef)**power
    gamma - from rbf kernel
    coef - bias term in polynomial kernel
    
    Kernel of SVC was coded by cvxopt(ConvexOptimization) library
    That's why i didnt want to realize it from scratch.
    Essential parts of this algorithm were realized by this library automatically
    Basically this realization is pointless!

    Learned realization and used code from this page:
    https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch
    /supervised_learning/support_vector_machine.py
    '''

    def __init__(self,C=1,kernel=rbf_kernel,power=4,gamma=None,coef=4):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vectors_labels = None
        self.intercept = None

    def fit(self,X,y):
        n_samples, n_features = np.shape(X)
        # Set gamma by default
        if not self.gamma:
            self.gamma = 1 / n_features
        # Initialize kernel method with parameters
        self.kernel = self.kernel(
                power=self.power,
                gamma=self.gamma,
                coef=self.coef)

        # Kernel Matrix
        kernel_matrix = np.zeros((n_samples,n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i,j] = self.kernel(X[i],X[j])

        # Define quadratic problem
        P = cvxopt.matrix(np.outer(y,y) * kernel_matrix,tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y,(1,n_samples),tc='d')
        b = cvxopt.matrix(0,tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max,G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max,h_min)))
        # Solve
        minimization = cvxopt.solvers.qp(P,q,G,h,A,b)

        # Langrage multipliers
        lagr_mult = np.ravel(minimization['x'])

        # Extract support vectors
        # Get indexes of non-zero lagr. multipliers
        idx = lagr_mult > 1e-7
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        # Get the samples that will act as support vectors
        self.support_vectors = X[idx]
        # Get the corresponding labels
        self.support_vector_labels = y[idx]

        # Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= (self.lagr_multipliers[i] * 
                    self.support_vector_labels[i] *
                    self.kernel(self.support_vectors[i],self.support_vectors[0]))

    def predict(self,X):
        y_pred = []
        # For each sample 
        for sample in X:
            prediction = 0
            # Classificate
            for i in range(len(self.lagr_multipliers)):
                prediction += (self.lagr_multipliers[i] * 
                        self.support_vector_labels[i] * 
                        self.kernel(self.support_vectors[i],sample))
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)


