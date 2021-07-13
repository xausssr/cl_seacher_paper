

# Дополнения к статье "Оценка гиперпараметров сверточных нейронных сетей для классификации объектов"

Данный репозиторий содержит код для воспроизведения экспериментов из указанной статьи.

Содержание репозитория:
  * `example_mnist_ru.ipynb` &minus; jupyter тетрадка с подробным примером использования предлагаемой методики
  * `clseacher.py` &minus; реализация методики в отдельной функции `search`

Использование:
```
...
import clseacher as cl
...

<загрузка обучающих данных>
<описание базовой модели>

cl.search(model, data, A=0.999, alpha=0.1, pvalue=0.005, print_iter=True, plot_search=True, lang='ru')
```


Окружение:
  * numpy      1.20.1
  * tensorflow 2.2.0
  * keras      2.3.0-tf
  * scipy      1.4.1
  * seaborn    0.10.0


# Code section for paper "Hyperparameter Estimation of Convolutionsl Neural Networks for Object Classification"

Repository items:
  * `example_mnist_eng.ipynb` &minus; jupyter notebook with a detailed example of using the proposed technique
  * `clseacher.py` &minus; implementation of the technique in a separate `search` function

Usage:
```
...
import clseacher as cl
...

<load training data>
<compile baseline model>

cl.search(model, data, A=0.999, alpha=0.1, pvalue=0.005, print_iter=True, plot_search=True, lang='eng')
```

Libraries:
  * numpy      1.20.1
  * tensorflow 2.2.0
  * keras      2.3.0-tf
  * scipy      1.4.1
  * seaborn    0.10.0
