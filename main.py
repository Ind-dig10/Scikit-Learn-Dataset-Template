import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib


# Информация об датасете
def info_dataset(dataset):
    dictionary_info = dict()

    # Размерность
    dictionary_info['Shape'] = dataset.shape

    # Общая информация
    dictionary_info['Info'] = dataset.info()

    # Получение средних значений
    dictionary_info['Describe'] = dataset.describe()

    # Кол-во самых встречаю
    dictionary_info['Nunique'] = dataset.nunique()

    # Группировка значений
    return dictionary_info


# Вывести значения колонки
# 1 параметр - dataset, 2 - Наименование колонки, 3 - Индекс
def info_column_dataset(dataset, column_name, index):
    return dataset[column_name][:index]


# Построить круговую диаграмму
def plot_dataset(dataset, column_name, index):
    dataset = get_counts_value_dataset(dataset, column_name)
    dataset[:index].plot.pie(figsize = (5, 5), autopct = '% 1.1f %%', startangle = 90)
    plt.show()


# Получить уникальные данные (тут мы узнаем что данные могут быть загрязнены)
def get_unigue_value(dataset, column_name):
    return dataset[column_name].unique()


# Получить часто встречающиеся значения по убыванию
def get_counts_value_dataset(dataset, column_name):
    return dataset[column_name].value_counts()


# Отбор числовых колонок
def get_numeric_colums(dataset):
    df_numeric = dataset.select_dtypes(include=[np.number])
    return df_numeric.columns.values


# Отбор нечисловых колонок
def get_non_numeric_columns(dataset):
    df_non_numeric = dataset.select_dtypes(exclude=[np.number])
    return df_non_numeric.columns.values


# Получить Тепловую карту пропущенных значений
# желтый - пропущенные данные, синий - не пропущенные
def get_heatmap_missing_values(dataset, index):
    cols = dataset.columns[:index]  # первые n колонок
    colours = ['#000099', '#ffff00']
    sns.heatmap(dataset[cols].isnull(), cmap=sns.color_palette(colours))
    plt.show()


# Получить список пропущенных значений в процентном соотношении
def get_list_percent_missing_values(dataset):
    for col in dataset.columns:
        pct_missing = np.mean(dataset[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing * 100)))


# Получить среднее значение столбцов
def get_avg_value_column_dataset(dataset, column_name):
    count = dataset[column_name].count()
    sum = dataset[column_name].sum()
    return sum/count


if __name__ == '__main__':
    dataset = pd.read_csv("Data/survey.csv")
    dataset.head()
    get_list_percent_missing_values(dataset)
    #print(info_column_dataset(dataset, 'Age', 10))
    #dict = info_dataset(dataset_survey)
    #plot_dataset(dataset, 'Age', 10)
    #get_list_percent_missing_values(dataset)

    #df = dataset.loc[(dataset['Age'] != 0) & (dataset['Age'] < 100)]
    #print(df.std())




