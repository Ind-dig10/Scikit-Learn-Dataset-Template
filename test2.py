import numpy as np
import pandas as pd
from scipy.cluster.vq import whiten
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
import scipy.stats
from scipy import stats
from sklearn.ensemble import IsolationForest
from scipy.stats import norm, skew #for some statistics
from scipy.stats import normaltest

from IPython.display import Image
from IPython.core.display import HTML

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc

import os
import warnings


def info_dataset(dataset):
    dataset.info()


def search_gap(dataset):
    return dataset.isnull().sum()


def describe_dataset(dataset):
    return dataset.describe()


def get_standard_deviation(dataset):
    return dataset.describe().iloc[2, :].pow(0.5)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def test():
    f, axes = plt.subplots(4, 4, figsize=(10, 10))
    sns.despine(left=True)
    for i in range(0, 16):
        sns.distplot(X["D_" + str(i)], fit=norm, hist=False, ax=axes[i // 4, i % 4])


def correlation():
    alpha = 0.05

    corr_y3_factors = np.zeros((20, 1), dtype=bool)
    for i in range(0, np.shape(X)[1]):
        if (stats.pearsonr(Y.iloc[:, 2], X.iloc[:, i])[1] < alpha):
            corr_y3_factors[i] = True
        else:
            corr_y3_factors[i] = False

    print(np.where(corr_y3_factors)[0])


data = pd.read_csv('Data/dataset.csv')
data = data.sample(frac=1)

#Преобразование во float
data1 = data.replace(',','.', regex=True)
data.iloc[:, 8:12] = data1.iloc[:, 8:12].astype(float)
data.iloc[:, 14:16] = data1.iloc[:, 14:16].astype(float)
data.iloc[:, 18:24] = data1.iloc[:, 18:24].astype(float)
X = data.iloc[:, 4:25]
Y = data.iloc[:, 1:4]

#sns.boxplot(data=X)
#f, axes = plt.subplots(2, 10, figsize=(30, 10))
#sns.despine(left=True)
#for i in range(0, 20):
#    sns.boxplot(data = X["X" + str(i+1)], ax=axes[i//10, i%10])


#f, axes = plt.subplots(2, 10, figsize=(30, 10))
#sns.despine(left=True)
#for i in range(0, 20):
#    sns.distplot(X["X" + str(i+1)], fit=norm, hist=False, ax=axes[i//10, i%10])

#plt.show()

#for i in range(1, 21):
#    m, mMinus, mPlus = mean_confidence_interval(X["X" + str(i)], confidence=0.95)
#    print("Доверительный интервал для X" + str(i) + " : Среднее: %f с интервалом(%f , %f)" % (m, mMinus, mPlus))


#for i in range(1, 20):
#    stat, p = normaltest(X["X" + str(i)])
#    print('Оценка нормальности распределения для X' + str(i) + '. Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
#    alpha = 0.05
#    if p > alpha:
#        print('Подчиняется нормальному закону распределения')
#    else:
#        print('Не подчиняется нормальному закону распределения')

def z_score(df):
    df_std = df.copy()
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()

    return df_std


#Тест сферичности барлетта
def get_bartlett(dataset):
    chi_square_value, p_value = calculate_bartlett_sphericity(dataset)
    print(chi_square_value, p_value)


#Тест Кайзера-Мейера-Олкина (КМО)
def get_kmo(dataset):
    kmo_all, kmo_model = calculate_kmo(dataset)
    print(kmo_model)


def test(dataset):
    fa = FactorAnalyzer(n_factors=25, rotation=None)
    fa.fit(dataset)
    ev, v = fa.get_eigenvalues()
    plt.scatter(range(1, dataset.shape[1] + 1), ev)
    plt.plot(range(1, dataset.shape[1] + 1), ev)
    plt.title('Scree Plot')
    plt.xlabel('Факторы')
    plt.ylabel('Собственное значение')
    plt.grid()
    plt.show()


def cluster_analysis(data):
    norm_data = whiten(data.iloc[:, 1:])
    norm_data = pd.DataFrame(data=norm_data)
    clusters = 3
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(norm_data)
    print(norm_data[kmeans.labels_ == 0])

    f, axes = plt.subplots(2, 10, figsize=(30, 10))
    sns.despine(left=True)
    cluster0 = norm_data[kmeans.labels_ == 0]
    cluster1 = norm_data[kmeans.labels_ == 1]
    cluster2 = norm_data[kmeans.labels_ == 2]
    for i in range(0, 20):
        ax = axes[i // 10, i % 10]
        a_heights, a_bins = np.histogram(cluster0.iloc[:, i + 3])
        b_heights, b_bins = np.histogram(cluster1.iloc[:, i + 3])
        c_heights, c_bins = np.histogram(cluster2.iloc[:, i + 3])

        width = (a_bins[1] - a_bins[0]) / 3

        ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
        ax.bar(b_bins[:-1] + width, b_heights, width=width, facecolor='seagreen')
        ax.bar(c_bins[:-1] + 2 * width, c_heights, width=width, facecolor='red')
        ax.set_title(label="X" + str(i + 1))
    plt.show()

#corrmat = data.corr()
#f, ax = plt.subplots(figsize=(18, 14))
#sns.heatmap(corrmat, square=True, annot = True);
norm_data = whiten(data.iloc[:, 1:])
norm_data = pd.DataFrame(data=norm_data)
clusters = 3

kmeans = KMeans(n_clusters=clusters)
kmeans.fit(norm_data)
norm_data[kmeans.labels_ == 0]

df_data_standardized = z_score(data)
#print(df_data_standardized)
#print(describe_dataset(data))
#get_bartlett(data)
#get_kmo(data)
cluster_analysis(data)