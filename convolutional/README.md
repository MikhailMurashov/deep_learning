# Convolutional

## [Введение](../README.md)

Вся проведенная работа содержится в Jupiter-ноутбуке [main.ipynb](main.ipynb). Файл включает в себя:

* Подготовку тестовых, тренировочных, валидационных данных
* Создание моделей
* Обучение моделей
* Визуализацию каждого шага
* Тестирование модели с использованием тестовых данных

## Теория

В наших моделях было использовано:

- Функция активации Relu на скрытых слоях: 

![relu](https://latex.codecogs.com/gif.latex?relu%28x%29%20%3D%20%5Cmax%20%280%2C%20x%29)

- Функция активации SoftMax на выходном слое: 

![softmax](https://latex.codecogs.com/gif.latex?softmax%28x%29%20%3D%20%5Cfrac%7Be%5E%7Bx%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BK%7De%5E%7Bx_%7Bk%7D%7D%7D)

- Слой BatchNormalization. Этот слой нормализует данные по батчу: сводит математическое ожидание к 0 и дисперсию к 1 ([arxiv](https://arxiv.org/abs/1502.03167))

- Слой Dropout, который случайным образом отключает часть нейронов. Это позволяет предотвратить переобучение, путем тренировки некого ансамбля сетей ([arxiv](https://arxiv.org/abs/1207.0580))

- Слой GlobalAveragePooling2D, который берет среднее по всем каналам. Это позволяет сильно уменьшить количество обучаемых параметров на классификационном Dense слое, что в свою очередь помогает бороться с переобучением

- Оптимизатор Adam - adaptive momentum ([arxiv](https://arxiv.org/abs/1412.6980v9)).  ***Learning rate=1e-2*** , установлен по умолчанию в выбранной библиотеке

![m](https://latex.codecogs.com/gif.latex?m_%7Bt%7D%20%3D%20%5Cbeta_1%20m_%7Bt-1%7D%20&plus;%20%281%20-%20%5Cbeta_1%29%20%5Cnabla%20f%28x_t%29)     
![v](https://latex.codecogs.com/gif.latex?v_%7Bt%7D%20%3D%20%5Cbeta_2%20v_%7Bt-1%7D%20&plus;%20%281%20-%20%5Cbeta_2%29%20%5Cnabla%20%28f%28x_t%29%29%5E2)

![hat_m](https://latex.codecogs.com/gif.latex?%5Cwidehat%7Bm_t%7D%20%3D%20%5Cfrac%7Bm_t%7D%7B1%20-%20%5Cbeta_%7B1%7D%5Et%7D)     
![hat_v](https://latex.codecogs.com/gif.latex?%5Cwidehat%7Bv_t%7D%20%3D%20%5Cfrac%7Bv_t%7D%7B1%20-%20%5Cbeta_%7B2%7D%5Et%7D)

![x](https://latex.codecogs.com/gif.latex?x_%7Bt&plus;1%7D%20%3D%20x_t%20-%20%5Calpha%20%5Cfrac%7B%5Cwidehat%7Bm_t%7D%7D%7B%5Csqrt%7B%5Cwidehat%7Bv_t%7D%7D%20&plus;%20%5Cvarepsilon%7D)

 - Функция ошибки ***categorical cross-entropy*** для классификации на несколько классов:

 ![crossentropy](https://latex.codecogs.com/gif.latex?categorical%5C_crossentropy%20%3D%20-%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20y_i%20%5Ccdot%20%5Clog%28%5Chat%7By_i%7D%29)

- Инициализация начальных весов с помощью Xavier uniform initializer, установлен по умолчанию в выбранной библиотеке

## Конфигурации моделей

В данной работе были проведены эксперементы со следующими моделями

1. ***Сверточная модель с тремя скрытыми слоями***

[//]: #![](models_img/size=2,lr=0.001,use_dropout=False,use_batchnorm=False,use_globalpool=False.png)

| Layer (type) | Output Shape   | Param #   |
|--------------|----------------|-----------|
| Conv2D(7x7x64,stride=2)       | (112, 112, 64) | 9472      |
| Relu         | (112, 112, 64) | 0         |
| MaxPooling2D | (56, 56, 64)   | 0         |
| Conv2D(3x3x128)      | (56, 56, 128)  | 73856     |
| Relu         | (56, 56, 128)  | 0         |
| Conv2D(3x3x128)       | (56, 56, 128)  | 147584    |
| Relu         | (56, 56, 128)  | 0         |
| MaxPooling2D | (28, 28, 128)  | 0         |
| Flatten      | (100352)       | 0         |
| Dense        | (256)          | 25690368  |
| Dense        | (37)           | 9509      |

Total params: 25,930,789

Training time: 255 s

![Model Accuracy](models_img/model1Acc.png)
![Model Loss](models_img/model1Loss.png)

Наблюдается переобучение, в связи с разношерстностью данных, о которой упоминалось в предыдущих работах.

2. ***Сверточная модель с использованием Batch Normalization + Global Pooling***

Добавим в модель слой GlobalAveragePooling2D для уменьшения количества обучаемых параметров. Также добавим слой BatchNormalization, чтобы проэкспементировать со скоростью сходимости модели.

[//]: #![](models_img/size=3,lr=0.001,use_dropout=False,use_batchnorm=True,use_globalpool=True.png)

| Layer (type)           |Output Shape    | Param #   |
|------------------------|----------------|-----------| 
| Conv2D(7x7x64,stride=2)                 | (112, 112, 64) | 9472      |
| batch_normalization    | (112, 112, 64) | 256       |
| Relu                   | (112, 112, 64) | 0         |
| max_pooling            | (56, 56, 64)   | 0         |
| Conv2D(3x3x128)                 | (56, 56, 128)  | 73856     |
| batch_normalization    | (56, 56, 128)  | 512       |
| Relu                   | (56, 56, 128)  | 0         |
| Conv2D(3x3x128)                 | (56, 56, 128)  | 147584    |
| batch_normalization    | (56, 56, 128)  | 512       |
| Relu                   | (56, 56, 128)  | 0         |
| max_pooling            | (28, 28, 128)  | 0         |
| Conv2D(3x3x256)                 | (28, 28, 256)  | 295168    |
| batch_normalization    | (28, 28, 256)  | 1024      |
| Relu                   | (28, 28, 256)  | 0         |
| Conv2D(3x3x256)                 | (28, 28, 256)  | 590080    |
| batch_normalization    | (28, 28, 256)  | 1024      |
| Relu                   | (28, 28, 256)  | 0         |
| max_pooling            | (14, 14, 256)  | 0         |
| global_average_pooling | (256)          | 0         |
| Dense                  | (37)           | 9509      |

Total params: 1,128,997

Training time: 598 s

![Model Accuracy](models_img/model2Acc.png)
![Model Loss](models_img/model2Loss.png)

Переобучение снизилось. Точность на тренировочной выборке возросла до 0,68. Сходится модель намного быстрее благодаря слоям батч нормализации. Попробуем добавить дропаут перед слоем классификации, чтобы еще снизить переобучение.

3. ***Сверточная модель с использованием Batch Normalization + Global Pooling + Dropout***

[//]: #![](models_img/size=3,lr=0.001,use_dropout=True,use_batchnorm=True,use_globalpool=True.png)

| Layer (type)           | Output Shape   | Param #   |
|------------------------|----------------|-----------| 
| Conv2D(7x7x64,stride=2)         | (112, 112, 64) | 9472      |
| batch_normalization    | (112, 112, 64) | 256       |
| Relu                   | (112, 112, 64) | 0         |
| max_pooling            | (56, 56, 64)   | 0         |
| Conv2D(3x3x128)        | (56, 56, 128)  | 73856     |
| batch_normalization    | (56, 56, 128)  | 512       |
| Relu                   | (56, 56, 128)  | 0         |
| Conv2D(3x3x128)        | (56, 56, 128)  | 147584    |
| batch_normalization    | (56, 56, 128)  | 512       |
| Relu                   | (56, 56, 128)  | 0         |
| max_pooling            | (28, 28, 128)  | 0         |
| Conv2D(3x3x256)        | (28, 28, 256)  | 295168    |
| batch_normalization    | (28, 28, 256)  | 1024      |
| Relu                   | (28, 28, 256)  | 0         |
| Conv2D(3x3x256)        | (28, 28, 256)  | 590080    |
| batch_normalization    | (28, 28, 256)  | 1024      |
| Relu                   | (28, 28, 256)  | 0         |
| max_pooling            | (14, 14, 256)  | 0         |
| global_average_pooling | (256)          | 0         |
| dropout(0,7)           | (256)          | 0         |
| Dense                  | (37)           | 9509      |

Total params: 1,128,997

Training time: 1445 s

![Model Accuracy](models_img/model3Acc.png)
![Model Loss](models_img/model3Loss.png)

Большой процент дропаута сильно помог с переубучением! Сохраним на состояние модели для последующих тестов.

#### Попробуем применить аугментацию наших изображений

- случайный поворот на 10 градусов
- случайное зуммирование на 10%
- случайное изменение яркости в диапозоне [0.7, 1.3]
- случайное горизонтальное отображение

Это должно привести к более стабильным результатам. Используем предыдущую архитектуру сети, т.к. она дала самые лучшие результаты.

Training time: 7326 s

![Model Accuracy](models_img/model4Acc.png)
![Model Loss](models_img/model4Loss.png)


Как видно из графика сильно это не помогло. Предположение: уменьшить learning rate, возможно модель зависла на плато и сойдется еще чуть-чуть. Используем lr=0.0003, как советовал ***Andrej Karpathy*** в своем [твиттере](https://twitter.com/karpathy/status/801621764144971776) =D

Training time: 1582s

![Model Accuracy](models_img/model5Acc.png)
![Model Loss](models_img/model5Loss.png)

***+12% точности*** на валидационной выборке! Неплохо. Остановимся на этом и сохраним модель.

## Тестирование моделей

| Model | loss   | accuracy |
|:------: |:------:|:--------:|
| 1 | 7.8739 | 0.1462   |
| 2 | 6.6024 | 0.1262   |
| 3 | 1.4370 | 0.6047   |
| 3* | 1.0759 |  0.6677   |

3* - 3-я модель применением аугментации

### Пример 

![Birman](models_img/Birdman.png)

Expected category : Birman

| Category | 1 | 2 | 3 | 3* |
|:--------:|:----------------:|:----------------:|:----------------:|:-------------:|
|Birman    | 0.6536           | 0.6022        |  0.2598           | 0.6058        |
|Siamese   | 0.3092           | 0.2023        |  0.7225           | 0.3518        |
|Ragdoll   | 0.0343           | 0.0921        |  0.0143           | 0.0421        |

3* - 3-я модель применением аугментации
