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
	''' Точность = tp/tp+fp,
	где tp - true positive, fp - false positive
	Например, целевое значение 0, определенное классификатором как 1 - 
	-  false positive, а целевое значение 1, 
	определенное как 1 - true positive.
	'''
	true_positive = 0
	false_positive = 0
	for real,predicted in zip(y,y_pred):
		if int(real) == 1 and int(predicted) == 1:
			true_positive += 1
		if int(real) == 0 and int(predicted) == 1:
			false_positive += 1
	# Считаем полноту
	return true_positive/(true_positive + false_positive)


def recall_score(y,y_pred):
	''' Полнота = tp/(tp+fn), где fn - false negative
	'''
	true_positive = 0
	false_negative = 0
	for real,predicted in zip(y,y_pred):
		if int(real) == 1 and int(predicted) == 1:
			true_positive += 1
		if int(real) == 1 and int(predicted) == 0:
			false_negative += 1
	# Считаем полноту 
	return true_positive/(true_positive + false_negative)


def f1_score(y,y_pred):
	''' F1-мера '''
	precision = precision_score(y,y_pred)
	recall = recall_score(y,y_pred)
	# Считаем F1-меру просто по формуле
	return 2 * (precision * recall)/(precision + recall)
	