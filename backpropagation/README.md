# Backpropagation on MNIST from scratch

## Цель работы

Цель настоящей работы состоит в том, чтобы изучить метод обратного распространения ошибки для обучения глубоких нейронных сетей на примере двуслойной полностью связанной сети (один скрытый слой).

## Задачи

Выполнение практической работы предполагает решение следующих задач:

1. Изучение общей схемы метода обратного распространения ошибки. 
2. Вывод математических формул для вычисления градиентов функции ошибки по параметрам нейронной сети и формул коррекции весов
3. Проектирование и разработка программной реализации. 
4. Тестирование разработанной программной реализации. 
5. Подготовка отчета, содержащего минимальный объем информации по каждому этапу выполнения работы. 

В процессе выполнения лабораторной работы предполагается, что сеть ориентирована на решение задачи классификации одноканальных изображений. Типичным примером такой задачи является задача классификации рукописных цифр. Именно ее предлагается использовать в качестве тестовой задачи на примере набора данных MNIST. 

Метод обратного распространения ошибки разрабатывается, исходя из следующих предположений:

1. На входе сети имеется *w×h* нейронов, что соответствует разрешению изображения. 
2. На выходе сети имеется *k* нейронов, что соответствует количеству классов изображений. 
3. Скрытый слой содержит *s* нейронов. 
4. В качестве функции активации на втором слое используется функция softmax. 
5. В качестве функции ошибки используется кросс-энтропия.

## Ход работы

1. Были разработаны две модели для классификации рукописных цифр с помощью библиотеки Keras.
	- Двухслойная полносвязная сеть
	- Сверточная сеть
2. Проведены эксперименты с реализованными моделями.
3. Был разработал класс ***Network*** состоящий из двухслойной полносвязной сети и реализации метода обратного распространения ошибки.
	- Модель состоит из:
		- Входного слоя (input_dim=728)
		- Скрытого слоя (кол-во нейронов=300)
		- Функции активации на скрытом слое ReLU
		- Выходного слоя (кол-во нейронов=10)
		- Фукции активации на выходном слое Softmax
	- Алгоритм стохастического градиентного спуска с использованием моментума и мини батчей

## Теория

 - Функция активации ***ReLU*** на скрытом слоях: 

![relu](https://latex.codecogs.com/gif.latex?relu%28x%29%20%3D%20%5Cmax%20%280%2C%20x%29)

 - Функция активации ***Softmax*** на выходном слое: 

![softmax](https://latex.codecogs.com/gif.latex?softmax%28x%29%20%3D%20%5Cfrac%7Be%5E%7Bx%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BK%7De%5E%7Bx_%7Bk%7D%7D%7D)

 - Функция ошибки ***cross-entropy*** для классификации на несколько классов:

![crossentropy](https://latex.codecogs.com/gif.latex?crossentropy%20%3D%20-%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%20%5Ccdot%20%5Clog%28%5Chat%7By_i%7D%29)

 - ***SGD with momentum*** помогает гасить разброс градиента:

![momentum](https://latex.codecogs.com/gif.latex?m_%7Bt&plus;1%7D%20%3D%20%5Cbeta%20m_t%20&plus;%20%281%20-%20%5Cbeta%29%5Cnabla%20f%28x_t%29)

![sgd](https://latex.codecogs.com/gif.latex?x_%7Bt&plus;1%7D%20%3D%20x_t%20&plus;%20%5Calpha%20m_%7Bt&plus;1%7D)

 - Инициализация весов производится случайными числами с дисперсией = 1\n, где n - это количество входных нейронов: `np.random.randn(h, n) * np.sqrt(1. / n)`

#### Вычисление градиента:

Дана двухслойая нейронная сеть:

![](https://latex.codecogs.com/gif.latex?x%20%3D%20input%5C%5C%20z%20%3D%20W_hx%20%5C%5C%20h%20%3D%20ReLU%28z%29%20%5C%5C%20%5Ctheta%20%3D%20W_oh%20%5C%5C%20%5Chat%7By%7D%20%3D%20Softmax%28%5Ctheta%29%20%5C%5C%20E%20%3D%20crossentropy%28y%2C%20%5Chat%7By%7D%29)

Вычислять производую будем с помощью правила ***chain rule***:

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20W_0%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%5Chat%7By%7D%7D%20%5Cfrac%7B%5Cpartial%20%5Chat%7By%7D%7D%7B%5Cpartial%20%5Ctheta%20%7D%20%5Cfrac%7B%5Cpartial%20%5Ctheta%7D%7B%5Cpartial%20W_0%7D)

Введем вспомогательные дельты:

![](https://latex.codecogs.com/gif.latex?%5Cdelta_1%20%3D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20%5Ctheta%7D%2C%20%5Cdelta_2%20%3D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20z%7D)

![](https://latex.codecogs.com/gif.latex?%5Cdelta_1%20%3D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20%5Ctheta%7D%20%3D%20%28%5Chat%7By%7D%20-%20y%29%5ET)

![](https://latex.codecogs.com/gif.latex?%5Cdelta_2%20%3D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20%5Ctheta%7D%20%5Cfrac%7B%5Cpartial%20%5Ctheta%7D%7B%5Cpartial%20h%7D%20%5Cfrac%7B%5Cpartial%20h%7D%7B%5Cpartial%20z%7D%20%3D%20%5Cdelta_1%20%5Cfrac%7B%5Cpartial%20%5Ctheta%7D%7B%5Cpartial%20h%7D%20%5Cfrac%7B%5Cpartial%20h%7D%7B%5Cpartial%20z%7D%20%3D%20%5Cdelta_1%20W_0%20%5Cfrac%7B%5Cpartial%20h%7D%7B%5Cpartial%20z%7D%20%3D%20%5Cdelta_1%20W_0%20%5Ccdot%20%7BReLU%7D%27%28z%29%3D%5Cdelta_1W_0%5Ccdot%20sign%28h%29)

Вычисляем производные:

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20W_0%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20%5Ctheta%7D%20%5Cfrac%7B%5Cpartial%20%5Ctheta%7D%7B%5Cpartial%20W_0%7D%20%3D%20%5Cdelta_1%20%5Cfrac%7B%5Cpartial%20%5Ctheta%7D%7B%5Cpartial%20W_0%7D%20%3D%20%5Cdelta_1%5ET%20h%5ET)

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20W_h%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20z%7D%20%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20W_h%7D%20%3D%20%5Cdelta_2%20%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20W_h%7D%20%3D%20%5Cdelta_2%5ET%20x%5ET)

## Результаты экспериментов

 - ***batch_size*** = 256
 - ***learning_rate*** = 0.1
 - ***beta*** = 0.9
 - ***epochs*** = 20

#### Лог обучения
```
Epoch 1: training loss = 0.35727803700688
Epoch 2: training loss = 0.293442041578463
Epoch 3: training loss = 0.25784482991381
Epoch 4: training loss = 0.23049475357331473
Epoch 5: training loss = 0.2079890471256404
Epoch 6: training loss = 0.18907099772309838
Epoch 7: training loss = 0.17319915395742091
Epoch 8: training loss = 0.15961626789623984
Epoch 9: training loss = 0.14791220445370057
Epoch 10: training loss = 0.13776174791057114
Epoch 11: training loss = 0.12889546971695237
Epoch 12: training loss = 0.12106654077423026
Epoch 13: training loss = 0.11412679392753333
Epoch 14: training loss = 0.1078735654324865
Epoch 15: training loss = 0.10227742797760349
Epoch 16: training loss = 0.09721577404338205
Epoch 17: training loss = 0.09259833966904699
Epoch 18: training loss = 0.08839015148286934
Epoch 19: training loss = 0.08449859016070198
Epoch 20: training loss = 0.08089822325730113
```

test loss = 0.11011547531721864

#### Тестирование разработанной модели

Метрики для оценки качества построенной модели:

![](https://latex.codecogs.com/gif.latex?precision%20%3D%20%5Cfrac%7BTP%7D%7BTP%20&plus;%20FP%7D%20%5C%5C%20recall%20%3D%20%5Cfrac%7BTP%7D%7BTP%20&plus;%20FN%7D%20%5C%5C%20F_%5Cbeta%20%3D%20%281%20&plus;%20%5Cbeta%5E2%29%20%5Ccdot%20%5Cfrac%7Bprecision%20%5Ccdot%20recall%7D%7B%28%5Cbeta%5E2%20%5Ccdot%20precision%29%20&plus;%20recall%7D)

| # | precision | recall | f1-score | support |
|---|:---------:|:------:|:--------:|:-------:|
| 0 | 0.98      | 0.99   | 0.98     | 693     |
| 1 | 0.98      | 0.97   | 0.98     | 758     |
| 2 | 0.97      | 0.97   | 0.97     | 716     |
| 3 | 0.96      | 0.97   | 0.97     | 689     |
| 4 | 0.97      | 0.95   | 0.96     | 723     |
| 5 | 0.94      | 0.97   | 0.95     | 594     |
| 6 | 0.98      | 0.98   | 0.98     | 697     |
| 7 | 0.97      | 0.96   | 0.97     | 729     |
| 8 | 0.96      | 0.96   | 0.96     | 689     |
| 9 | 0.94      | 0.96   | 0.95     | 712     |

total accuracy  0.97

### Тестирование моделей построенных в библиотеке Keras

#### Полносвязная модель

 - SGD, learning_rate=0.1, momentum=0.9
 - batch_size=256
 - epochs=10
 - инициализация весов Xavier uniform (uniform distribution `[-limit, limit]`, where `limit = sqrt(6 / (fan_in + fan_out))`, where `fan_in = number of input`, `fan_out = number of output`)

| Layer (type) | Output Shape   | Param #   |
|--------------|----------------|-----------|
|Flatten  |       (784)    |           0      |   
|Dense    |       (300)     |          235500 |   
|Dense    |       (10)      |          3010   |   

Total params: 238,510

```
Epoch 1/10
 - 3s - loss: 0.3180 - accuracy: 0.9070 - val_loss: 0.1529 - val_accuracy: 0.9552
Epoch 2/10
 - 2s - loss: 0.1279 - accuracy: 0.9640 - val_loss: 0.1013 - val_accuracy: 0.9699
Epoch 3/10
 - 2s - loss: 0.0885 - accuracy: 0.9745 - val_loss: 0.0845 - val_accuracy: 0.9749
Epoch 4/10
 - 2s - loss: 0.0674 - accuracy: 0.9803 - val_loss: 0.0790 - val_accuracy: 0.9751
Epoch 5/10
 - 2s - loss: 0.0528 - accuracy: 0.9845 - val_loss: 0.0787 - val_accuracy: 0.9758
Epoch 6/10
 - 2s - loss: 0.0425 - accuracy: 0.9874 - val_loss: 0.0677 - val_accuracy: 0.9786
Epoch 7/10
 - 2s - loss: 0.0344 - accuracy: 0.9906 - val_loss: 0.0646 - val_accuracy: 0.9800
Epoch 8/10
 - 2s - loss: 0.0292 - accuracy: 0.9920 - val_loss: 0.0611 - val_accuracy: 0.9809
Epoch 9/10
 - 2s - loss: 0.0226 - accuracy: 0.9944 - val_loss: 0.0623 - val_accuracy: 0.9800
Epoch 10/10
 - 2s - loss: 0.0192 - accuracy: 0.9955 - val_loss: 0.0655 - val_accuracy: 0.9796
```

Test loss: 0.0654

Test accuracy: 0.9796

#### Сверточная модель

 - Adam, learning_rate=0.001
 - batch_size=128
 - epochs=5
 - инициализация весов Xavier uniform

| Layer (type) | Output Shape   | Param #   |
|--------------|----------------|-----------|
| Conv2D           | (26, 26, 32) |  320     |   
| Conv2D          | (24, 24, 64) |  18496   |   
| MaxPooling2D | (12, 12, 64) |  0       |   
| Dropout         | (12, 12, 64) |  0       |   
| Flatten          | (9216)       |  0       |   
| Dense            | (128)        |  1179776 |   
| Dropout         | (128)        |  0       |   
| Dense             | (10)         |  1290    |   

Total params: 1,199,882

```
Epoch 1/10
 - 136s - loss: 0.2330 - accuracy: 0.9281 - val_loss: 0.0506 - val_accuracy: 0.9836
Epoch 2/10
 - 137s - loss: 0.0816 - accuracy: 0.9761 - val_loss: 0.0406 - val_accuracy: 0.9866
Epoch 3/10
 - 139s - loss: 0.0613 - accuracy: 0.9812 - val_loss: 0.0328 - val_accuracy: 0.9886
Epoch 4/10
 - 137s - loss: 0.0516 - accuracy: 0.9846 - val_loss: 0.0309 - val_accuracy: 0.9899
Epoch 5/10
 - 139s - loss: 0.0458 - accuracy: 0.9857 - val_loss: 0.0297 - val_accuracy: 0.9905
```

Test loss: 0.0298

Test accuracy: 0.9901

## Заключение

В рамках выполненной работы достигнуты поставленные цели:

1. Подготовлено пошаговое описание метода обратного распространения ошибки с выводом всех математических формул для сети, описанной в разделе Задачи. 
2. Разработана программная реализация метода для рассматриваемого частного случае. 
3. Разработано приложение для решения задачи классификации рукописных цифр на примере базы MNIST. 
4. Подготовлены результаты классификации для тестового набора данных MNIST. 
5. Результаты классификации удовлетворяют поставленным условиям (~ 97%)
