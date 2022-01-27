import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from warnings import filterwarnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


def no_of_cluster(df, title):
    w=[]
    e=[]
    for i in range(1,10):
        k=KMeans(n_clusters=i)
        k.fit_predict(df)
        e.append(k.inertia_)
        w.append(i)
    plt.figure(figsize=(8,5))
    plt.plot(w,e,'bo-')
    plt.title(f"Optimum number of Clusters for KMeans - {title}")


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


def get_age(dataset):
    age = []
    for i in dataset.Age:
        if (i < 18) or (i > 99):
            age.append(31)  # Median
        else:
            age.append(i)

    dataset['Age'] = age

    other = ['A little about you', 'p', 'Nah', 'Enby', 'Trans-female', 'something kinda male?', 'queer/she/they',
             'non-binary', 'All', 'fluid', 'Genderqueer', 'Androgyne', 'Agender', 'Guy (-ish) ^_^',
             'male leaning androgynous', 'Trans woman', 'Neuter', 'Female (trans)', 'queer',
             'ostensibly male, unsure what that really means', 'trans']
    male = ['male', 'Male', 'M', 'm', 'Male-ish', 'maile', 'Cis Male', 'Mal', 'Male (CIS)', 'Make', 'Male ', 'Man',
            'msle', 'cis male', 'Cis Man', 'Malr', 'Mail']
    female = ['Female', 'female', 'Cis Female', 'F', 'f', 'Femake', 'woman', 'Female ', 'cis-female/femme',
              'Female (cis)', 'femail', 'Woman', 'female']

    dataset['Gender'].replace(to_replace=other, value='other', inplace=True)
    dataset['Gender'].replace(to_replace=male, value='M', inplace=True)
    dataset['Gender'].replace(to_replace=female, value='F', inplace=True)

    print('\033[1m' + 'Уникальные значения в столбце "Пол"  :' + '\033[0m', dataset.Gender.unique())
    print('\033[1m' + 'Диапазон возраста :' + '\033[0m', (dataset.Age.min(), dataset.Age.max()))


def test(df):
    df_ = df.drop(['Age', 'Country'], axis=1)

    buttons = []
    i = 0
    vis = [False] * 24

    for col in df_.columns:
        vis[i] = True
        buttons.append({'label': col,
                        'method': 'update',
                        'args': [{'visible': vis},
                                 {'title': col}]})
        i += 1
        vis = [False] * 24

    fig = go.Figure()

    for col in df_.columns:
        fig.add_trace(go.Pie(
            values=df_[col].value_counts(),
            labels=df_[col].value_counts().index,
            title=dict(text='Distribution of {}'.format(col),
                       font=dict(size=18, family='monospace'),
                       ),
            hole=0.5,
            hoverinfo='label+percent', ))

    fig.update_traces(hoverinfo='label+percent',
                      textinfo='label+percent',
                      textfont_size=12,
                      opacity=0.8,
                      showlegend=False,
                      marker=dict(colors=sns.color_palette('YlGn').as_hex(),
                                  line=dict(color='#000000', width=1)))

    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                      updatemenus=[dict(
                          type='dropdown',
                          x=1.15,
                          y=0.85,
                          showactive=True,
                          active=0,
                          buttons=buttons)],
                      annotations=[
                          dict(text="<b>Choose<br>Column<b> : ",
                               showarrow=False,
                               x=1.06, y=0.92, yref="paper", align="left")])

    for i in range(1, 22):
        fig.data[i].visible = False

    fig.show()


def test_2(df):
    fig = make_subplots(rows=2, cols=1)

    fig.append_trace(go.Bar(
        y=df['Country'].value_counts(),
        x=df['Country'].value_counts().index,
        name='Observations from Countries (Log)',
        text=df['Country'].value_counts(),
        textfont=dict(size=10,
                      family='monospace'),
        textposition='outside',
        marker=dict(color="#6aa87b")
    ), row=1, col=1)

    fig.append_trace(go.Histogram(
        x=df['Age'],
        nbinsx=8,
        text=['16', '500', '562', '149', '26', '5', '1'],
        marker=dict(color="#6aa87b")),
        row=2, col=1)

    # For Subplot : 1

    fig.update_xaxes(
        row=1, col=1,
        tickfont=dict(size=10, family='monospace'),
        tickmode='array',
        ticktext=df['Country'].value_counts().index,
        tickangle=60,
        ticklen=6,
        showline=False,
        showgrid=False,
        ticks='outside')

    fig.update_yaxes(type='log',
                     row=1, col=1,
                     tickfont=dict(size=15, family='monospace'),
                     tickmode='array',
                     showline=False,
                     showgrid=False,
                     ticks='outside')

    fig.update_traces(
        marker_line_color='black',
        marker_line_width=1.2,
        opacity=0.6,
        row=1, col=1)

    fig.update_xaxes(range=[-1, 48], row=1, col=1)


    fig.update_xaxes(
        title=dict(text='Age',
                   font=dict(size=15,
                             family='monospace')),
        row=2, col=1,
        tickfont=dict(size=15, family='monospace', color='black'),
        tickmode='array',
        ticktext=['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79'],
        ticklen=6,
        showline=False,
        showgrid=False,
        ticks='outside')

    fig.update_yaxes(
        row=2, col=1,
        tickfont=dict(size=15, family='monospace'),
        tickmode='array',
        showline=False,
        showgrid=False,
        ticks='outside')

    fig.update_traces(
        marker_line_color='black',
        marker_line_width=2,
        opacity=0.6,
        row=2, col=1)

    fig.update_layout(height=1200, width=900,
                      title=dict(
                          text='Одномерная визуализация переменных  1. Наблюдения из стран (Журнал) 2. Подсчет возрастов',
                          x=0.5,
                          font=dict(size=16, color='#27302a',
                                    family='monospace')),
                      plot_bgcolor='#edf2c7',
                      paper_bgcolor='#edf2c7',
                      showlegend=False)

    fig.show()


def test_3(df):
    df.treatment = df.treatment.astype('category')
    df.treatment = df.treatment.cat.codes
    df.treatment.value_counts()

    X = df.drop('treatment', axis=1)
    y = df.treatment

    cols = X.columns

    encoder = LabelEncoder()
    for col in cols:
        encoder.fit(X[col])
        X[col] = encoder.transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

    filterwarnings("ignore")

    steps = [('scaler', StandardScaler()),
             ('log_reg', LogisticRegression())]

    pipeline = Pipeline(steps)

    parameters = dict(log_reg__solver=['newton-cg', 'lbfgs', 'liblinear'],
                      log_reg__penalty=['l2'],
                      log_reg__C=[100, 10, 1.0, 0.1, 0.01])

    cv = GridSearchCV(pipeline,
                      param_grid=parameters,
                      cv=5,
                      scoring='accuracy',
                      n_jobs=-1,
                      error_score=0.0)

    cv.fit(X_train, y_train)
    y_pred = cv.predict(X_test)
    log_accuracy = accuracy_score(y_pred, y_test) * 100

    print('\033[1m' + 'Лучшие показатели : ' + '\033[0m', cv.best_params_)
    print('\033[1m' + 'Точность : {:.2f}%'.format(log_accuracy) + '\033[0m')
    print('\033[1m' + 'Отчет показателей классификации : ' + '\033[0m\n', classification_report(y_test, y_pred))

    cm = confusion_matrix(y_pred, y_test)
    print('\033[1m' + 'Матрица ошибок : ' + '\033[0m')
    plt.figure(dpi=100)
    sns.heatmap(cm, cmap='YlGn', annot=True, fmt='d')
    plt.show()


if __name__ == '__main__':
    dataset = pd.read_csv("Data/russian_demography.csv")
    dataset.head()
    print(get_counts_value_dataset(dataset, 'npg'))
    #get_list_percent_missing_values(dataset)
    #print(info_column_dataset(dataset, 'Age', 10))
    #dict = info_dataset(dataset_survey)
    #plot_dataset(dataset, 'Age', 10)
    #get_list_percent_missing_values(dataset)
    #print('\033[1m' + 'Всего исследованных стран :' + '\033[0m', len(dataset.Country.value_counts()))
    #print('\033[1m' + 'Уникальные :' + '\033[0m\n', dataset.state.unique())

   # dataset['self_employed'] = dataset['self_employed'] \
   #     .fillna(pd.Series(np.random.choice(['Yes', 'No'], p=[0.117647, 0.882353], size=len(dataset))))

   # dataset['work_interfere'] = dataset['work_interfere'] \
    #    .fillna(pd.Series(np.random.choice(['Sometimes', 'Never', 'Rarely', 'Often']
     #                                      , p=[0.467337, 0.214070, 0.173869, 0.144724], size=len(dataset))))

    #print('\033[1m' + 'Total empty values in the Dataset :' + '\033[0m', dataset.isnull().sum().sum())
    #df = dataset.loc[(dataset['Age'] != 0) & (dataset['Age'] < 100)]
    #print(df.std())
    #get_age(dataset)
    #test(dataset)
    #test_2(dataset)
    #test_3(dataset)



