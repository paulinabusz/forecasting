import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/Paulinka/Desktop/Paulina/Coronavirus dataset/2019_nCoV_data.csv", delimiter = ",")

#print(data.head())

### data cleansing ###

#check null values
#print(data.isnull().sum()) #there is 311 records with no value for Province/State 

#check datatype
#print(data.dtypes)

#removing unnecesary column
data.drop(['Sno'], axis = 1, inplace = True)

#country standaralizing
data=data.replace(to_replace ="China", 
                 value ="Mainland China")

#converting to date
data['Date'] = pd.to_datetime(data['Date'])
data['Last Update'] = pd.to_datetime(data['Last Update'])


Province = data['Province/State'].values.tolist()
Province = set(Province)
Province = list(Province)


Conutry = data['Country'].values.tolist()
Conutry = set(Conutry)
Conutry = list(Conutry)

#latest data selection

from datetime import date
latest_data = data[data['Last Update'] > pd.Timestamp(date(2020,1,31))]


### data overview ###

latest_data_stats = latest_data.groupby(['Country','Province/State']).sum()
print(latest_data_stats)

overall_deaths = data.groupby(['Country'])['Deaths'].sum()
print(overall_deaths)

overall_cases = data.groupby(['Country'])['Confirmed'].sum()
print(overall_cases)


### basic analysis ###

#top 5 countries (the highest death rate)
Country_cases=data.groupby(["Country"]).sum().reset_index()
top_5_countries = Country_cases.nlargest(5,['Confirmed']).reset_index(drop=True)
#reset_index(drop=True) - starting with nex indexing
#print(top_5_countries)

print(top_5_countries)
#bar_plot = plt.bar(top_5_countries['Country'], top_5_countries['Confirmed'])
#Bar plot is unreadable due to large number of confirmed cases in China - divide China by regions to see the spread among conutry

#Region_cases=data.groupby(["Province/State"]).sum().reset_index()
#Region_china_cases=data.groupby(["Province/State"]).sum().reset_index()
#print(Region_cases)
#top_5_china_regions = Region_china_cases.nlargest(5,['Confirmed']).reset_index(drop=True)
#bar_plot = plt.bar(top_5_china_regions['Province/Region'], top_5_china_regions['Confirmed'])

### Model for China ###

# Chinese data
Country_index = data.set_index("Country")
print(Country_index)
Chinese_data = Country_index.xs("Mainland China")
print(Chinese_data)