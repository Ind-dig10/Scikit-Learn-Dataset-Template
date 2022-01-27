# data analysis
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

import cufflinks
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot



# детальный анализ
def detailed_analysis(df):
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ratio = (df.isnull().sum() / obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt()
    print('Data shape:', df.shape)

    cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis']
    details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis], axis=1,
                        sort=False)

    details.columns = cols
    dtypes = details.types.value_counts()
    print('____________________________\nData types:\n', dtypes)
    print('____________________________')
    return details


def dataset_info(dataset):
    return (dataset.iloc[:, 1:].describe())


def population_decline(dataset):
    return dataset.groupby('year')[['birth_rate', 'death_rate']].mean()


def create_diagram(dataset):
    cufflinks.go_offline()
    pd = population_decline(dataset)
    pd.iplot(mode='lines+markers', xTitle='Year', yTitle='Average',
                title='Yearly Average birth_rate and death_rate')


if __name__ == '__main__':
    dataset = pd.read_csv("Data/russian_demography.csv")
    dataset.head()
    detailed_analysis(dataset)
    #details = detailed_analysis(dataset)
    #print(population_decline(dataset))
