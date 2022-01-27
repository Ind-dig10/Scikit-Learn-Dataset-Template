from scripts.Dataset import Dataset
import numpy as np
import scipy.stats
from scipy import stats
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten, kmeans


class TemplateData(Dataset):

    #Конструктор
    def __init__(self, dataframe):
        self.df = dataframe
        data1 = self.df.replace(',', '.', regex=True)
        self.df.iloc[:, 8:12] = data1.iloc[:, 8:12].astype(float)
        self.df.iloc[:, 14:16] = data1.iloc[:, 14:16].astype(float)
        self.df.iloc[:, 18:24] = data1.iloc[:, 18:24].astype(float)
        self.X = self.df.iloc[:, 4:25]
        self.Y = self.df.iloc[:, 1:4]

    # функция для вычисления доверительного интервала
    def mean_confidence_interval(self, confidence=0.95):
        a = 1.0 * np.array(self.df)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m - h, m + h

    def get_mean_confidence_interval(self):
        for i in range(1, 21):
            m, mMinus, mPlus = self.mean_confidence_interval(self.X["X" + str(i)])
            print("Доверительный интервал для X" + str(i) + " : Среднее: %f с интервалом(%f , %f)" % (m, mMinus, mPlus))

    def brt(self):
        chi_square_value, p_value = calculate_bartlett_sphericity(self.df)
        return chi_square_value, p_value

    def gfdg(self):
        kmo_all, kmo_model = calculate_kmo(self.X)
        return kmo_model

    def factorad(self):
        fa = FactorAnalyzer(n_factors=25, rotation=None)
        fa.fit(self.X)
        # Check Eigenvalues
        ev, v = fa.get_eigenvalues()
        pd.DataFrame(ev)


    def kmeans(self, clusters = 3):
        norm_data = whiten(self.df.iloc[:, 1:])
        norm_data = pd.DataFrame(data=norm_data)
        self.clusters = clusters
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(norm_data)

    def linear_regression(self,clusters = 3, max_ind_lab_s=None):
        norm_data = whiten(self.df.iloc[:, 1:])
        norm_data = pd.DataFrame(data=norm_data)
        self.clusters = clusters
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(norm_data)
        y = self.df.iloc[:, 1:4]
        x = self.df.iloc[:, 4:]
        x = StandardScaler().fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        # Данные только для 2 кластера
        yCL2 = self.df[kmeans.labels_ == max_ind_lab_s].iloc[:, 1:4]
        xCL2 = self.df[kmeans.labels_ == max_ind_lab_s].iloc[:, 4:]
        xCL2 = StandardScaler().fit_transform(xCL2)
        xCL2_train, xCL2_test, yCL2_train, yCL2_test = train_test_split(xCL2, yCL2, test_size=0.3, random_state=0)
        Y_train = y_train.iloc[:, 2]
        Y_test = y_test.iloc[:, 2]
        lr_y3 = LogisticRegression(multi_class="auto")
        lr_y3.fit(x_train, Y_train)
        print("Уравнение линейной регресии:")
        y_string = "y = 0"
        for i in range(np.shape(lr_y3.coef_[0])[0]):
            y_string += str(round(lr_y3.coef_[0][i], 4)) + "*x" + str(i) + " + "
        y_string += str(round(lr_y3.intercept_[0], 4));
        print(y_string)




