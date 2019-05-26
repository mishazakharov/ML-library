import numpy as np
from math import sqrt


def mean_squared_error(y,y_pred):
	''' 
	Возвращает среднеквадратичную ошибку между величинами
	как в sklearn ЭТО НЕ RMSE
	'''
	m = y_pred.shape[0]
	return (1/m)*np.sum((y-y_pred)**2)


def mean_absolute_error(y,y_pred):
	'''
	Возвращает абсолютную ошибку.
	ЗДЕСЬ ТОЖЕ НЕТ КОРНЯ!(ВСЕ ПРАВИЛЬНО)
	'''
	m = y_pred.shape[0]
	# mean = 1/m
	# 1/m*np.sum можно заменить np.average, тоже самое!
	return (1/m)*np.sum(np.abs(y - y_pred))


def median_absolute_error(y,y_pred):
	''' Median!!! '''
	# median = среднее число
	return np.median(np.abs(y - y_pred))


def mean_squared_log_error(y,y_pred):
	''' БЕЗ КОРНЯ, как и в sklearn 
	'''
	# log1p = log(1+x)
	# в формуле стоит такой логарифм!
	y = np.log1p(y)
	y_pred = np.log1p(y_pred)
	# Так же реализовано и в sklearn
	return mean_squared_error(y,y_pred)


def accuracy_score(y,y_pred):
	''' metrics.accuracy_score '''
	# Счетчик совпадений
	right = 0
	# Считаем сколько предсказаний оказалось правильными
	for first,second in zip(y,y_pred):
		if int(first) == int(second):
			right += 1
	# Возвращаем процент правильных предсказаний
	return right/int(y_pred.shape[0])


def precision_score(y,y_pred):
	''' Полнота '''
	raise NotImplementedError


def f1_score(y,y_pred):
	''' F1-мера '''
	raise NotImplementedError