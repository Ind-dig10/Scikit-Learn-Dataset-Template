import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Dataset:

    # Конструктор
    def __init__(self, dataframe):
        self.df = dataframe
        self.shape = dataframe.shape
        self.describe = dataframe.describe()
        self.nunique = dataframe.nunique()


    # Общая информация
    def get_info(self):
        return self.df.info()

    @property
    def Shape(self):
        return self.shape

    @property
    def Describe(self):
        return self.describe

    @property
    def Nunique(self):
        return self.nunique

    # вывести значение колонки
    def get_info_column(self, column_name, index):
        return self.df[column_name][:index]

    # Получить часто встречающиеся значения по убыванию
    def get_counts_value_dataset(self, column_name):
        return self.df[column_name].value_counts()

    # Построить круговую диаграмму
    def plot_dataset(self, column_name, index):
        dataset = self.get_counts_value_dataset(column_name)
        dataset[:index].plot.pie(figsize=(5, 5), autopct='% 1.1f %%', startangle=90)
        plt.show()

    # Отбор числовых колонок
    def get_numeric_colums(self):
        df_numeric = self.df.select_dtypes(include=[np.number])
        return df_numeric.columns.values

    # Отбор нечисловых колонок
    def get_non_numeric_columns(self):
        df_non_numeric = self.df.select_dtypes(exclude=[np.number])
        return df_non_numeric.columns.values

    # Получить уникальные данные (тут мы узнаем что данные могут быть загрязнены)
    def get_unigue_value(self, column_name):
        return self.df[column_name].unique()

    # Получить часто встречающиеся значения по убыванию
    def get_counts_value_dataset(self, column_name):
        return self.df[column_name].value_counts()

    # Получить Тепловую карту пропущенных значений
    # желтый - пропущенные данные, синий - не пропущенные
    def get_heatmap_missing_values(self, index):
        cols = self.df.columns[:index]  # первые n колонок
        colours = ['#000099', '#ffff00']
        sns.heatmap(self.df[cols].isnull(), cmap=sns.color_palette(colours))
        plt.show()

    # Получить список пропущенных значений в процентном соотношении
    def get_list_percent_missing_values(self):
        for col in self.df.columns:
            pct_missing = np.mean(self.df[col].isnull())
            print('{} - {}%'.format(col, round(pct_missing * 100)))

    # Получить среднее значение столбцов
    def get_avg_value_column_dataset(self, column_name):
        count = self.df[column_name].count()
        sum = self.df[column_name].sum()
        return sum / count