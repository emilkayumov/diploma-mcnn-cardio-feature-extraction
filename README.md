# report-spring-2017
Этот репозиторий создан для отчёта перед научным руководителем о текущей деятельности. Проведённые эксперименты, прочитанные статьи, мысли, планы, идеи. Ниже этой шапки свежие записи выше, старые утекают вниз.

**14.02.2017**

_Просмотрел в последнее время_:
1. Classification of ECG Signal by Using Wavelet Transform and SVM (Zahra Golrizkhatami), 2015:
* с помощью вейвлетов находят все нужные элементы на кардиограмме (QRS-комплекс) и по ним строят признаки

- используют датасет MIT-BIH (48 записей, разные болезни сердца)
- средняя точность выявления болезней около 97 процентов

2. ECG Beats Classification Using Mixture of Features (Manab Kumar Das, Samit Ari), 2014:
- признаки по пикам, фурье и вейвлетам
- MIT-BIH, при этом тестируют они на тех людях, только на других участках записей

И другие статьи. В них либо про ручное выделение участков, либо по преобразованию Фурье, вейлетами, фильтрами. Есть статьи про сравнение основных подходов к преобразованиям. Подробно не читал. 

_Чем занимаюсь_:

Экспериментирую с сетями, навеянными [Multi-Scale Convolutional Neural Networks](https://arxiv.org/abs/1603.06995), о которых услышал на хакатоне, где классифицировали EEG (24 канала, а не 1). Казалось бы, здравая мысль брать одновременно сигналы разных длин (один и тот же участок сжатый в разное число раз позволяет в разных масштабах смотреть на сигнал), гонять по ним свёртки и брать от этого глобальный пулинг, чтобы избавиться от каких-либо локальных признаков. И должны получаться обученные находить свойственные заболеваниям места фильтры. Но ничего дельного пока не вышло. 

Дальше планирую пробовать автокодировщики в надежде получить на внутреннем представлении признаки для другого алгоритма. 

Ещё хочется поработать с преобразованиями Фурье, но при этом не усреднять их, делая 1D, а оставить в виде 2D. Тогда можно рассматривать сигнал в виде условно картинки (а это навеяно рассказом Дмитрия Ульянова про перенос стиля для музыки, где одномерный сигнал переводят вот так и работают с таким представлением).

_Что хочу и как вижу:_

Возможно, дело в моих ограниченных представлениях, но мне первым и единственным в голову приходит обучить сеть некоторого подходящего вида (возможно, для классификации, а можно автокодировщик или ещё что угодно на выходе), чтобы использовать внутреннее представление как набор признаков для какого-то классического алгоритма. И показать, что вот такое построение признаков (автоматическое) показывает более высокие финальные результаты, чем другие признаки, не требующие эвристик (например, из преобразования Фурье). 

_Ещё:_

Работаю пока со старым датасетом. Новый посмотрел, частично прогонял старый код по выявлению ИБС. Некоторые эвристики, кажется, ломаются (эвристики ведь). Какие-то результаты падают, какие-то растут (AUC вверх, точность вниз, например). В ближайшее время проделаю нормально и перееду на новый датасет. 
