import numpy as np
from random import randrange
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn


class LinearRegression():
	''' Класс линейной регрессии. '''

	def __init__(self,learning_rate=0.00001,number_of_iterations=3000):
		# Инициализация переменных
		self.learning_rate = learning_rate
		self.number_of_iterations = number_of_iterations
		self.theta = None

	def step_gradient(self,old_theta,vector_x,vector_y):
		# Количество сэмплов
		m = len(vector_y)
		# Вектор параметров
		theta = old_theta
		# Вектор дэльта(появляется при векторизации)
		delta = np.zeros((vector_x.shape[0],1))
		for i in range(0,m):
			# Определение векторов x и y
			x_i = vector_x
			y_i = vector_y
			x_i = x_i[:,i].reshape(vector_x.shape[0],1)
			y_i = y_i[i]
			# Гипотеза
			hypothesis = np.dot(theta.T,x_i)
			constant = (1/m) * (float(hypothesis) - y_i)
			delta += (x_i) * constant
			# Обновление весов
			new_theta = old_theta - (self.learning_rate * delta)

		return new_theta

	def gradient_runner(self,inital_theta,vector_x,vector_y):
		# Функция обучения модели
		theta = inital_theta
		for i in range(self.number_of_iterations):
			theta = self.step_gradient(theta,vector_x,vector_y)

		return theta

	def fit(self,X,y):
		# Матрица обучающей выборки n+1 x m размерная
		x_i = X.T
		# Добавление x0 = 1
		vector_x = np.insert(x_i,0,[1],axis=0)
		vector_y = y
		# "Случайная" инициализация весов!
		inital_theta = np.zeros((vector_x.shape[0],1))
		self.theta = self.gradient_runner(inital_theta,vector_x,vector_y)

	def predict(self,x):
		# Создание списка, хранящего результаты
		predictions = []
		# Определние размера полученного массива
		m = x.shape[0]
		# Предсказывает по одному сэмплу, добавляя все предсказания в список
		for i in range(0,m):
			x_i = x[i]
			# Добавление x0 = 1
			x_i = np.insert(x_i,0,[1],axis=0)
			# Предсказание 
			prediction = np.dot(self.theta.T,x_i)
			predictions.append(prediction)
		# Составление вектора пр ответов!
		predictions = np.array(predictions).reshape(-1,1)

		return predictions

	def get_weights():
		# Функция, возвращающая вектор весов модели!
		return theta


class LogisticRegression():
	''' Класс логистической регрессии. '''
	def __init__(self,learning_rate=0.00159,number_of_iterations=3000):

		self.learning_rate = learning_rate
		self.number_of_iterations = number_of_iterations
		self.theta = None

	def step_gradient(self,old_theta,vektor_x,vektor_y):
		m = len(vektor_y)
		# Вектор тэта
		theta = old_theta
		# Вектор дэльта начальный
		delta = np.zeros((vektor_x.shape[0],1))
		for i in range(0,m):  
			# Создание вектора n+1 РАЗМЕРНОСТИ(ДОБАВИЛ x0!)
			x_i = vektor_x
			y = vektor_y
			# Н-размерный вектор, содержащий все признаки для 1 экземпляра!
			x_i = x_i[:,i].reshape((vektor_x.shape[0],1))
			# Число - целевое значение на m-ом экземпляре!
			y_i = y[i]
			# Аргумент функции гипотезы!
			z = np.dot(theta.T,x_i)
			# Гипотеза логистической регрессии
			hypothesis = 1/(1+np.exp(-z))
			# Расчет констант, почему-то иначе не работает!!!!
			constant = (1/m) * (float(hypothesis) - y_i)
			# Новый вектор дельта!
			delta += (x_i) * constant
		# Формула шага градиентного спуска в векторной форме!!!
		new_theta = old_theta - (self.learning_rate * delta)
		return new_theta
		# При вызове функции в качестве old_theta будет передаваться new_theta
		# предыдущего шага, так осуществляется правильное обновление вектора тэта.

	def gradient_runner(self,initial_theta,vektor_x,vektor_y):
		theta = initial_theta
		for i in range(self.number_of_iterations):
			theta = self.step_gradient(theta,vektor_x,vektor_y)
		return theta

	def fit(self,X,y):
		x_i = X.T
		vektor_x = np.insert(x_i,0,[1],axis=0)
		vektor_y = y
		initial_theta = np.zeros((vektor_x.shape[0],1))
		self.theta = self.gradient_runner(initial_theta,vektor_x,vektor_y)

	def predict(self,x):
		predictions = []
		m = x.shape[0]
		for i in range(0,m):
			# Формируем вектор предсказаний!
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


class SVM():
	''' Метод опроных векторов. '''
	def __init__(self):
		raise NotImplementedError


# ЭТОТ КЛАСС ОТНОСИТСЯ К ДЕРЕЬВЕЯМ ПРИНЯТИЯ РЕШЕНИЙ!
class Question():
    ''' Вопрос используется для разделение данных.
    Класс записывает номер колонки и связанное с ним значение.
    Метод "match" задает вопрос и возвращает True,если ответ "да".
    '''
    def __init__(self,column,value):
        self.column = column
        self.value = value

    def match(self,example):
        # Сравнивает значение признака в example со значением признака в
        # вопросе(фактически, задает вопрос)
        b = DecisionTree()
        val = example[self.column]
        if b.is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # Вспомогательный метод, выводящий вопрос в читаемом формате
        condition = '=='
        if is_numeric(self.value):
            condition = '>='
        return 'Is %s %s %s?' % (
            header[self.column],condition,str(self.value))

# Это тоже!!!
class Leaf():
    ''' Листовой узел классифицирует данные. Leaf - лист
    Хранит словарь с ключами-классами и значениями, показывающими
    сколько раз этот класс встречался в данных, дошедших до листового узла
    '''
    def __init__(self,rows):
    	b = DecisionTree()
    	self.predictions = b.class_counts(rows)

class Decision_Node():
    ''' Decision Node задает вопрос.
    Содержит ссылку на вопрос и на два дочерних узла
    '''
    def __init__(self,question,true_branch,false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree():
	''' Дерево принятя решений. '''
	def __init__(self):
		pass


	def unique_vals(self,rows,col):
		'''Возвращает количество уникальных элементов колонки'''
		return set([row[col] for row in rows])

	def class_counts(self,rows):
	    '''
	    Считает количество экземпляров обучающей выборки классов
	    Возвращает словарь, где ключ - класс, а значение - количество
	    '''
	    counts = {}
	    for row in rows:
	        label = row[-1]
	        if label not in counts:
	            counts[label] = 0 
	        counts[label] += 1
	    return counts

	def is_numeric(self,value):
	    ''' Возвращает True, если входные данные - число, иначе False '''
	    return isinstance(value,int) or isinstance(value,float)

	def partition(self,rows,question):
	    ''' Разделяет данные.
	    Для каждой строки в данных
	    '''
	    true_rows,false_rows = [], []
	    for row in rows:
	        if question.match(row):
	            true_rows.append(row)
	        else:
	            false_rows.append(row)
	    return true_rows,false_rows

	def gini(self,rows):
	    ''' Считает индекс Джинни. '''
	    counts = self.class_counts(rows)
	    impurity = 1
	    for lbl in counts:
	        prob_of_lbl = counts[lbl] / float(len(rows))
	        impurity -= prob_of_lbl ** 2
	    return impurity

	def information_gain(self,left,right,current_uncertainty):
	    ''' Считает увеличение информации.
	    Неопределенность начального узла минус взвешенные неопределенности 
	    двух дочерних узлов
	    '''
	    p = float(len(left)) / (len(left) + len(right))
	    return current_uncertainty - p * self.gini(left) - (1-p) * self.gini(right)

	def find_best_split(self,rows):
	    ''' Находит наилучший вопрос с помощью перебора каждого атрибута и его
	        значения, рассчитывая при этом увеличение информации.
	    '''
	    best_gain = 0 # хранит лучшее значение inf_gain
	    best_question = None # хранит лучши вопрос
	    current_uncertainty = self.gini(rows) # неопределенность начального узла
	    n_features = len(rows[0]) - 1 # количество колонок(признаков) - 1
	    for col in range(n_features): # для каждого признака
	        values = set([row[col] for row in rows]) # хранит уникальные значения
	        for val in values: # для каждого значения признака
	            question = Question(col,val) 
	            # разделяет данные, основываясь на текущем вопросе
	            true_rows,false_rows = self.partition(rows,question)
	            # если данные не разделяются этим вопросом,
	            # то пропускае это значение признака
	            if len(true_rows) == 0 or len(false_rows) == 0:
	                continue
	            # вычисляем увеличение информации после разделения по вопросу
	            gain = self.information_gain(true_rows,false_rows,
	            								current_uncertainty)
	            # обновляется лучший gain и лучший question
	            if gain >= best_gain:
	                best_gain,best_question = gain,question
	    return best_gain,best_question


	def fit(self,X,y):
	    ''' Строит дерево.
	    '''
	    # Получаем rows. СЕЙЧАС ЛЕНЬ ПЕРЕПИСЫВАТЬ КОД ПОД НУЖНЫЙ API!!!!!
	    rows = np.c_[X,y]
	    # находим лучший вопрос и лучшее увеличение
	    gain,question = self.find_best_split(rows)
	    # если увеличение 0, то мы не можем больше задавать вопросы, поэтому
	    # возвращаем лист. (Базовый случай реукрсивной функции)
	    if gain == 0:
	        return Leaf(rows)
	    # если мы дошли до сюда, то мы нашли полезный атрибут/значение
	    # с помощью которого мы будем разделять данные
	    true_rows,false_rows = self.partition(rows,question)
	    true_X = np.delete(true_rows,-1,1)
	    true_y = np.array(true_rows)[:,-1]
	    false_X = np.delete(false_rows,-1,1)
	    false_y = np.array(false_rows)[:,-1]
	    # рекурсивно создаем true branch
	    true_branch = self.fit(true_X,true_y)
	    # рекурсивно создаем false branch
	    false_branch = self.fit(false_X,false_y)
	    # Возвращаем узел вопроса(Question Node)
	    # Записывает лучший атрибут/значение и каким ветвям следовать(true/false)
	    return Decision_Node(question,true_branch,false_branch)

	def classify(self,row,node):
	    # Базовый случай, мы достигли листа
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


'''
Наследование классов в DecisionTree не нужно!!!
Просто создать экземпляры вне класса и вызывать их в класс!!! 
намного проще!!!!!!
'''


    

class RandomForest():
	''' Случайный лес. '''
	def __init__(self,n_estimators=3,criterion='gini',max_depth=None):
		''' Инициализация переменных, не все реализованы! '''
		self.n_estimators = n_estimators
		self.criterion = criterion 
		self.max_depth = max_depth
		self.predictions = []
		self.trees = []

	def subsample(self,dataset_x,dataset_y, ratio):
		''' Функция, прверащающая исходный training_data
		в обучающий поднабор. Размер зависит от коэффицента ratio!
		Индексы выбираются случайно каждый раз.
		'''
		dataset = np.c_[dataset_x,dataset_y]
		sample = list()
		n_sample = round(len(dataset) * ratio)
		while len(sample) < n_sample:
			index = randrange(len(dataset))
			sample.append(dataset[index])
		return sample

	def fit(self,X,y):
		''' Построение случайного леса. '''
		
		for i in range(self.n_estimators):
			# Подготовка данных (Бэггинг)
			new_data = np.array(self.subsample(X,y,0.5))
			# Так как new_data возвращает пробэггиный датасет с икс и игрек
			# Поэтому ниже разделяем его на new_y и new_X
			new_y = new_data[:,-1].reshape(-1,1)
			new_X = np.delete(new_data,-1,1)
			tree = DecisionTree()
			# ДА, НАСТОЛЬКО УЖАСНАЯ РЕАЛИЗАЦИЯ ИНТЕРФЕЙСА У ЭТОГО КЛАССА!!!!
			b = tree.fit(new_X,new_y)
			self.trees.append(b)


	def predict(self,predict_on_X):
		''' Предсказания случайного лсеа!
			Предсказания всего ансамбля деревьев будут получатся следующим
			образом - создаем пустой np.array подходящего размера,
			суммируем предсказания всех деревьев и делим вектор на 
			их количество и получаем предсказание всего ансамбля!!!! 
		'''
		conclusion = np.zeros((predict_on_X.shape[0],1))
		tree = DecisionTree()
		for b in self.trees:
			prediction = tree.predict(predict_on_X,b)
			print(prediction.shape)
			conclusion =  conclusion + prediction

		conclusion = conclusion / int(len(self.trees))
		# Возвращаем округленные значения вектора!
		return np.round(conclusion)	

	

class Bagging():
	''' Анасмаблевый метод обучения - баггинг. '''
	def __init__(self):
		raise NotImplementedError




class AdaBoostClassifier():
	''' Ансамблевый метод обучения - бустинг(AdaBoost)
		Классифицирует -1 и 1!
		В моих классах еще не реализован параметр sample_weigth,
		поэтому при создании этого класса придется использовать 
		sklearn модели...
	'''
	def __init__(self,n_estimators=50,base_estimator=''):
		self.n_estimators = n_estimators
		self.base_estimator = base_estimator

	def fit(self,X,Y):
		# Список, хранящий классификаторы
		self.models = []
		# Альфа - это вес каждого классификатора
		self.alphas = []
		# N - количество экземпляров в датасете
		N = len(X)
		# Вектор весов, ищначально инициализированный 1/N
		W = np.ones(N)/N
		for m in range(self.n_estimators):
			if self.base_estimator == '':
				# Стаданартной моделью используется дерево принятие решений
				tree = DecisionTreeClassifier(max_depth=1)
				tree.fit(X,Y,sample_weight=W)
				P = tree.predict(X)
			err = W.dot(P != Y)
			alpha = 0.5 * (np.log(1-err) - np.log(err))
			# Vecotrized form
			W = W*np.exp(-alpha*Y*P)
			# Нормализуем
			W = W/W.sum()
			self.models.append(tree)
			self.alphas.append(alpha)

	def predict(self,X):
		''' Класс предсказаний. '''
		N = len(X)
		FX = np.zeros(N)
		for alpha,tree in zip(self.alphas,self.models):
			FX += alpha*tree.predict(X)
		return np.sign(FX)#, FX



class StackingRegression():
	''' Ансамблевый метод - стэкинг!
	Пока реализовано только создание несходных моделей путем обучения
	разными алгоритмами!
	Мета-регрессором всегда выступает модель линейной регрессии(пока что!)
	Позже эту опцию можно будет выставлять самому!!
	'''
	def __init__(self,list_of_models):
		# Список с моделями ансамбля
		self.list_of_models = list_of_models
		# Список для хранения предсказаний на наборе валидации
		self.predictions_on_valid = []
		# Список для хранения предсказаний на испытательном наборе
		self.predictions_on_test = []


	def fit(self,X,y):

		train_X,valid_X,train_y,valid_y = train_test_split(X,y,test_size=0.5)
		for model in self.list_of_models:
			# Предсказания для набора валидации
			model.fit(train_X,train_y)
			prediction = model.predict(valid_X)
			self.predictions_on_valid.append(prediction)
			
		# Просто чтобы достать valid_y для новой модели в методе predict
		# со всеми так!
		self.valid_y = valid_y
		self.train_X = train_X
		self.train_y = train_y


	def predict(self,X):
		''' Что нужно исправить:
		Если использовать sklearn.linear_model.LinearRegression,
		то среднеквадратичная ошибка предсказаний уменьшается в разы!!!!
		'''

		# Предсказания на испытательном наборе!
		for model in self.list_of_models:
			model.fit(self.train_X,self.train_y)
			prediction_on_test = model.predict(X)
			self.predictions_on_test.append(prediction_on_test)

		# Преобразуем список с предсказаниями модели на испытательном наборе
		# в np.array
		new_test = np.zeros((self.predictions_on_test[0].shape[0],1))
		for prediction in self.predictions_on_test:
			new_test = np.c_[new_test,prediction]
		# Удаляем колонку нулей
		X_test = np.delete(new_test,0,1)
		# Мета-регрессором всегда выступает линейная регрессия(сейчас!)
		# Если использовать sklearn.linear_model.LinearRegression,
		# то среднеквадратичная ошибка предсказаний уменьшается в разы!!!!
		stacking_model = sklearn.linear_model.LinearRegression()
		new = np.zeros((self.predictions_on_valid[0].shape[0],1))
		# Преобразуем списк с предсказаниями модели на наборе валидации
		# в np.array!
		for prediction in self.predictions_on_valid:
			new = np.c_[new,prediction]
		# Удаляем колокнку нулей
		new_X = np.delete(new,0,1)
		stacking_model.fit(new_X,self.valid_y.reshape(-1,1))
		prediction_on_test = stacking_model.predict(X_test)

		return prediction_on_test



class VotingClassifier():
	''' Обычынй классифиатор голосвания! '''
	def __init__(self,estimators):
		# Список из моделей
		self.estimators = estimators
		# Храним фиты всех моделей
		self.models = []


	def fit(self,X,y):
		for each_model in self.estimators:
			fit = each_model.fit(X,y)
			self.models.append(fit)


	def predict(self,X):
		# 
		predictions = np.zeros((X.shape[0],1))
		for model in self.models:
			prediction = model.predict(X).reshape(-1,1)
			predictions += prediction
		# Делим на количество моделей
		predictions = predictions / len(self.models)
		function = lambda x: round(x)
		# Округляем значения!
		predictions = np.round(predictions)

		return predictions



class BaggingClassifier():
	''' Классификатор на баггинге '''
	def __init__(self,base_estimator='',n_estimators=10):
		self.base_estimator = base_estimator
		self.n_estimators = n_estimators
		# Список с обученными моделями
		self.list_of_trained_estimators = []

	def subsample(self,X,y,ratio):
		''' Функция, прверащающая исходный training_data
		в обучающий поднабор. Размер зависит от коэффицента ratio!
		Индексы выбираются случайно каждый раз.
		'''
		sample_X = list()
		sample_y = list()
		n_sample = round(len(X) * ratio)
		while len(sample_X) < n_sample:
			index = randrange(len(X))
			sample_X.append(X[index])
			sample_y.append(y[index])

		return sample_X,sample_y

	def fit(self,X,y):
		# Базовая модель - дерево принятия решений
		if self.base_estimator == '':
			self.base_estimator = DecisionTreeClassifier()
		# Разделяем данные, чтобы, используя одинаковые алгоритмы обчуения
		# получить несходные модели(они будут совершать ошибки разного рода)
		ratio = 1/self.n_estimators
		for i in range(self.n_estimators):
			new_X,new_y = self.subsample(X,y,ratio)
			b = self.base_estimator.fit(new_X,new_y)
			self.list_of_trained_estimators.append(b)

	def predict(self,X):
		predictions = np.zeros((X.shape[0],1))
		for model in self.list_of_trained_estimators:
			predictions += model.predict(X).reshape(-1,1)
		predicitons = predictions / self.n_estimators
		predictions = np.round(predicitons)

		return predictions


class CrossEntropy():
	def __init__(self):
		pass

	def mist(self,y,p):
		# Не делим на 0
		p = np.clip(p,1e-15,1-1e-15)
		return - y*np.log(p) - (1-y)*np.log(1-p)

	def gradient(self,y,p):
		# Не делим на 0
		p = np.clip(p,1e-15,1-1e-15)
		return -(y/p) + (1-y) / (1-p)

class MSE():
	def __init__(self):
		pass

	def mist(self,y,y_pred):
		return 0.5 * (y-y_pred)**2

	def gradient(self,y,y_pred):
		return -(y-y_pred)


class GradientBoosting():
	''' Супер-класс градиентного бустинг (на деревьях)'''
	def __init__(self,n_estimators,learning_rate,
				min_samples_split,min_impurity,max_depth,regression):
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.min_samples_split = min_samples_split
		self.min_impurity = min_impurity
		self.max_depth = max_depth
		self.regression = regression
		self.loss = MSE()
		if not self.regression:
			self.loss = CrossEntropy()

		# Инициализируем деревья
		self.trees = []
		for _ in range(self.n_estimators):
			tree = sklearn.tree.DecisionTreeRegressor(
								min_samples_split=self.min_samples_split,
								min_impurity_split=self.min_impurity,
								max_depth=self.max_depth)
			self.trees.append(tree)

	def fit(self,X,y):
		# Изначально берется среднее значение
		y_pred = np.full(np.shape(y),np.mean(y,axis=0)) 
		for tree in self.trees:
			# Градиент выбранной функции издержек
			# Базовая функция издержек - 1/2 MSE
			gradient = self.loss.gradient(y,y_pred)
			tree.fit(X,gradient)
			update = tree.predict(X)
			# Learning_rate введен, чтобы неменого регуляризовать модель
			y_pred -= self.learning_rate*update

	def predict(self,X):
		# Способ формирования предиктов у градиентного бустинга
		y_pred = np.array([])
		for tree in self.trees:
			update = tree.predict(X)
			update = (self.learning_rate*update)
			y_pred = -update if not y_pred.any() else y_pred - update

		if not self.regression:
			# Случай для классификации
			y_pred = np.exp(y_pred)/np.expand_dims(np.sum(np.exp(y_pred),axis=1),axis=1)
			y_pred = np.argmax(y_pred,axis=1)

		return y_pred



class GradientBoostingClassifier(GradientBoosting):
	''' Градиентный бустинг для классификации
	Просто используется другая функция издержки и другой способ предикта
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
	''' Регрессия, используя градиентный бустинг. '''
	def __init__(self,n_estimators=200,learning_rate=0.5,min_samples_split=2,
												min_impurity=1e-7,max_depth=4):
		# Наследует от супер-класса
		super(GradientBoostingRegressor,self).__init__(
						n_estimators=n_estimators,
						learning_rate=learning_rate,
						min_samples_split=min_samples_split,
						min_impurity=min_impurity,
						max_depth=max_depth,
						regression=True)

			





		




