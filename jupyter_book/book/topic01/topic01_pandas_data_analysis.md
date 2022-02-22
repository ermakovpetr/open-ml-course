---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(topic01)=

# Тема 1. Первичный анализ данных

```{figure} /_static/img/ods_stickers.jpg
:name: ods_stickers
```

**<center>[open ML course](https://ods.ai/tracks/open-ml-course) – открытый курс ODS по машинному обучению** </center><br>

Авторы: [Yury Kashnitsky](https://yorko.github.io). Translated and edited by [Christina Butsko](https://www.linkedin.com/in/christinabutsko/), [Yuanyuan Pao](https://www.linkedin.com/in/yuanyuanpao/), [Anastasia Manokhina](https://www.linkedin.com/in/anastasiamanokhina), Sergey Isaev and [Artem Trunov](https://www.linkedin.com/in/datamove/). This material is subject to the terms and conditions of the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.


```{figure} /_static/img/pandas.jpg
:width: 444px
:name: pandas
```

`Pandas` -- это библиотека Python, предоставляющая широкие возможности для анализа данных. Данные, с которыми работают датасаентисты, часто хранятся в форме табличек -- например, в форматах .csv, .tsv или .xlsx. С помощью библиотеки `Pandas` такие табличные данные очень удобно загружать, обрабатывать и анализировать с помощью SQL-подобных запросов. А в связке с библиотеками `Matplotlib` и `Seaborn` `Pandas` предоставляет широкие возможности визуального анализа табличных данных.


Основными структурами данных в `Pandas` являются классы `Series` и `DataFrame`. Первый из них представляет собой одномерный индексированный массив данных некоторого фиксированного типа. Второй – это двухмерная структура данных, представляющая собой таблицу, каждый столбец которой содержит данные одного типа. Можно представлять её как словарь объектов типа `Series`. Структура `DataFrame` отлично подходит для представления реальных данных: строки соответствуют признаковым описаниям отдельных объектов, а столбцы соответствуют признакам.

```{code-cell} ipython3
import pandas as pd
import numpy as np
from pathlib import Path
```

Будем показывать основные методы в деле, анализируя [набор данных](https://bigml.com/user/francisco/gallery/dataset/5163ad540c0b5e5b22000383) по оттоку клиентов телеком-оператора (скачивать не нужно, он есть в репозитории). Прочитаем данные (метод read_csv) и посмотрим на первые 5 строк с помощью метода `head`:

```{code-cell} ipython3
PATH_TO_DATA = Path("../../_static/data")

df = pd.read_csv(PATH_TO_DATA / "telecom_churn.csv")
df.head()
```

<details>
<summary>Printing DataFrames in Jupyter notebooks</summary>
<p>
In Jupyter notebooks, Pandas DataFrames are printed as these pretty tables seen above while `print(df.head())` is less nicely formatted.
By default, Pandas displays 20 columns and 60 rows, so, if your DataFrame is bigger, use the `set_option` function as shown in the example below:

```{code-cell} ipython3
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
```
</p>
</details>
<br>

# Интерактивные графики

```{code-cell} ipython3

import plotly.io as pio
import plotly.express as px
import plotly.offline as py

fig = px.scatter(df, x="Total day minutes", y="Customer service calls", color="Churn")
fig
```

```{note}
Это заметка. Формула Рамануджана: $\frac{1}{\pi}=\frac{2 \sqrt{2}}{99^{2}} \sum_{k=0}^{\infty} \frac{(4 k) !}{k !^{4}} \frac{26390 k+1103}{396^{4 k}}$
```


```{warning}
И так далее
```

## Ресурсы

* The same notebook as an interactive web-based [Kaggle Kernel](https://www.kaggle.com/kashnitsky/topic-1-exploratory-data-analysis-with-pandas)
* ["Merging DataFrames with pandas"](https://nbviewer.jupyter.org/github/Yorko/mlcourse.ai/blob/main/jupyter_english/tutorials/merging_dataframes_tutorial_max_palko.ipynb) -- a tutorial by Max Plako within mlcourse.ai (full list of tutorials is [here](https://mlcourse.ai/tutorials))
* ["Handle different dataset with dask and trying a little dask ML"](https://nbviewer.jupyter.org/github/Yorko/mlcourse.ai/blob/main/jupyter_english/tutorials/dask_objects_and_little_dask_ml_tutorial_iknyazeva.ipynb) -- a tutorial by Irina Knyazeva within mlcourse.ai
* Main course [site](https://mlcourse.ai), [course repo](https://github.com/Yorko/mlcourse.ai), and YouTube [channel](https://www.youtube.com/watch?v=QKTuw4PNOsU&list=PLVlY_7IJCMJeRfZ68eVfEcu-UcN9BbwiX)
* Official Pandas [documentation](http://pandas.pydata.org/pandas-docs/stable/index.html)
* Course materials as a [Kaggle Dataset](https://www.kaggle.com/kashnitsky/mlcourse)
* Medium ["story"](https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-1-exploratory-data-analysis-with-pandas-de57880f1a68) based on this notebook
* If you read Russian: an [article](https://habrahabr.ru/company/ods/blog/322626/) on Habr.com with ~ the same material. And a [lecture](https://youtu.be/dEFxoyJhm3Y) on YouTube
* [10 minutes to pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)
* [Pandas cheatsheet PDF](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)
* GitHub repos: [Pandas exercises](https://github.com/guipsamora/pandas_exercises/) and ["Effective Pandas"](https://github.com/TomAugspurger/effective-pandas)
* [scipy-lectures.org](http://www.scipy-lectures.org/index.html) -- tutorials on pandas, numpy, matplotlib and scikit-learn
