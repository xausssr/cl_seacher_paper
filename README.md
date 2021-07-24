

# Дополнения к статье "Оценка гиперпараметров сверточных нейронных сетей для классификации объектов"

Данный репозиторий содержит код для воспроизведения экспериментов из указанной статьи.

Содержание репозитория:
  * `example_mnist_ru.ipynb` &minus; jupyter тетрадка с подробным примером использования предлагаемой методики
  * `example_use_class_ru.ipynb` &minus; jupyter тетрадка с подробным примером использования класса `clseacher`
  * `clseacher.py` &minus; реализация методики в отдельной функции `search`

Использование:
```
...
import clseacher as cl
...

<загрузка обучающих данных>
<описание базовой модели>

seacher = cl(model, x_train.shape[1:], keras.metrics.AUC, lang='ru')
seacher.search(
    (x_train, y_train, x_test, y_test),
    epochs = 15,
    batch_size = 2048,
    save_path_folder = "example_path",
    eq_type = True, 
    A = 0.999,
    alpha = 0.1,
    p_value = 0.005,
    print_iter = True
)
```
Параметры конструктора класса:
  * `model` базовая модель, `keras.Sequential`
  * `input_shape` форма входного тензора
  * `metric` метрика, за которой будет следить функция поиска, `keras.metrics.Metric`
  * `lang` язык вывода (`ru` - русский, `eng` - английский)

Параметры метода `search`:
  * `data` - кортеж из 4 сущностей: обучающих данных, обучающей разметки, тестовых данных, тестовой разметки
  * `epoch` - количетсво эпох обучения на итерацию
  * `batch_size` - размер батча на обучении/тесте
  * `save_path_folder` - папка для сохранения моделей (на каждой итерации поиска сохраняется отдельная обученная модель)
  * `eq_type` - тип слежения за метрикой
     * True - остановка при метрики **меньшей** чем `A`
     * False - остановка при метрике **большей** чем `A`
  * `A` - пороговое значение метрики
  * `alpha` - пороговое значение косинусного расстояния
  *  `p_value` - p-значение в тесте на нормальность (критерий Пирсона)
  *  `print_iter` - вывод информации об итерации в процессе поиска

Окружение:
  * numpy      1.20.1
  * tensorflow 2.2.0
  * keras      2.3.0-tf
  * scipy      1.4.1
  * seaborn    0.10.0


# Code section for paper "Hyperparameter Estimation of Convolutionsl Neural Networks for Object Classification"

Repository items:
  * `example_mnist_eng.ipynb` &minus; jupyter notebook with a detailed example of using the proposed technique
  * `example_use_class_eng.ipynb` &minus; jupyter notebook with a detailed example of using the `clseacher` class
  * `clseacher.py` &minus; implementation of the technique in a separate `search` function

Usage:
```
...
import clseacher as cl
...

<load training data>
<compile baseline model>

seacher = cl(model, x_train.shape[1:], keras.metrics.AUC, lang='eng')
seacher.search(
    (x_train, y_train, x_test, y_test),
    epochs = 15,
    batch_size = 2048,
    save_path_folder = "example_path",
    eq_type = True, 
    A = 0.999,
    alpha = 0.1,
    p_value = 0.005,
    print_iter = True
)
```

Class constructor parameters:
  * `model` baseline, `keras.Sequential`
  * `input_shape` model input shape
  * `metric` metric to be monitored by the search function, `keras.metrics.Metric`
  * `lang` language (`ru` - Russian, `eng` - English)

`search` method parameters:
  * `data` - a tuple of 4 entities: training data, training labels, test data, test labels
  * `epoch` - количетсво эпох обучения на итерацию
  * `batch_size` - training/test batch size
  * `save_path_folder` - folder for saving models (a separate trained model is saved at each search iteration)
  * `eq_type` - type of tracking the metric 
     * True - stop when the metric **is less** than `A`
     * False - stop when the metric **is greater** than ` A`
  * `A` - metric threshold
  * `alpha` - cosine distance threshold
  *  `p_value` - p-value in the test for normality (Pearson's test)
  *  `print_iter` - printing iteration information during the search process
  
Libraries:
  * numpy      1.20.1
  * tensorflow 2.2.0
  * keras      2.3.0-tf
  * scipy      1.4.1
  * seaborn    0.10.0
