import matplotlib.pyplot as plt
import pandas as pd

class RussianDemography:
    # Конструктор
    def __init__(self, dataframe):
        self.df = dataframe


    def get_info_urbanization(self):
        data = self.df
        data_capital = pd.DataFrame()
        data_capital = data[(data['region'] == 'Moscow') | (data['region'] == 'Saint Petersburg') | (
                    data['region'] == 'Leningrad Oblast') | (data['region'] == 'Moscow Oblast')]
        data_capital = data_capital.drop(columns=['npg', 'birth_rate', 'death_rate', 'gdw'])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data_capital['year'][(data['region'] == 'Moscow')],
                data_capital['urbanization'][(data['region'] == 'Moscow')], ':b', label='Москва')
        ax.plot(data_capital['year'][(data['region'] == 'Moscow Oblast')],
                data_capital['urbanization'][(data['region'] == 'Moscow Oblast')], '-g', label='Московская область');
        ax.plot(data_capital['year'][(data['region'] == 'Saint Petersburg')],
                data_capital['urbanization'][(data['region'] == 'Saint Petersburg')], 'o', label='Санкт Петербург')
        ax.plot(data_capital['year'][(data['region'] == 'Leningrad Oblast')],
                data_capital['urbanization'][(data['region'] == 'Leningrad Oblast')], label='Ленинградская область')
        plt.legend()
        plt.show()

    def natural_population_growth_decline(self):
        self.df.groupby(['year']).agg({'birth_rate': 'mean', 'death_rate': 'mean'}).plot.bar(figsize=(20, 10),
            title='График естественного прироста и убили населения с 1990 по 2020')
        plt.show()


    # Получить топ областей с самой высокой рождаемостью
    def get_regions_highest_fertility(self):
        dfTopBirth = self.df.groupby(['region']).agg({'birth_rate': 'mean', 'death_rate': 'mean'}).sort_values(
            by="birth_rate", ascending=False).head(10)
        print(dfTopBirth)