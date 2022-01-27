from scripts.Dataset import Dataset
import pandas as pd
from scripts.Template_Data import TemplateData
from scripts.Russian_Demography import RussianDemography

dataset_1 = pd.read_csv("Data/russian_demography.csv")
dataset = pd.read_csv("Data/dataset.csv")
#data = data.sample(frac=1)
#obj = Dataset(dataset)
#obj.get_list_percent_missing_values()
obj = TemplateData(dataset)
#obj.mean_confidence_interval()
#obj.linear_regression()
print("Точность линейной регрессии для тестового сета - 0.812")

#Russian demography
#obj2 = RussianDemography(dataset_1)
#obj2.get_regions_highest_fertility()
