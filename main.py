import pandas as pd
import numpy as np


# Информация об датасете
def info_dataset(dataset):
    dictionary_info = dict()

    # Размерность
    dictionary_info['Shape'] = dataset.shape

    # Общая информация
    dictionary_info['Info'] = dataset.info()

    # Получение средних значений
    dictionary_info['Describe'] = dataset.describe()

    # Кол-во уникальных значений для каждого столбца
    dictionary_info['Nunique'] = dataset.nunique()
    return dictionary_info


if __name__ == '__main__':
    dataset_survey = pd.read_csv("Data/survey.csv")
    dataset_survey.head()
    dict = info_dataset(dataset_survey)
    print(dict['Info'])
