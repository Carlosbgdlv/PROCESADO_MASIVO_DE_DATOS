## FUNCIONES DEL NOTEBOK DE SERIES TEMPORALES:
#------------------------------------------------------------------

#Importamos las librerías


import pandas as pd
from pandas import datetime as dt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.api import tsa

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

#------------------------------------------------------------------

##Función nº 1
def preprocesado_pais(covid_vaccine_data,pais):
    """
    covid_vaccine_data: dataframe con todas las vacunas
    pais: pais que queremos escoger para analizar
    """
    vacc_pais = covid_vaccine_data.loc[covid_vaccine_data.country == pais]
    #print(pais)
    vacc_pais.set_index('date',inplace=True)
    
    
    #crear una funcion de preprocesado
    vacc_pais['daily_vaccinations']= vacc_pais['daily_vaccinations'].interpolate(method='cubicspline')
    vacc_pais['total_vaccinations']= vacc_pais['total_vaccinations'].interpolate(method='cubicspline')
    vacc_pais['people_vaccinated']= vacc_pais['people_vaccinated'].interpolate(method='cubicspline')

    vacc_pais['people_fully_vaccinated']= vacc_pais['people_fully_vaccinated'].interpolate(method='cubicspline')
    vacc_pais['total_vaccinations_per_hundred']= vacc_pais['total_vaccinations_per_hundred'].interpolate(method='cubicspline')
    vacc_pais['people_vaccinated_per_hundred']= vacc_pais['people_vaccinated_per_hundred'].interpolate(method='cubicspline')

    vacc_pais['people_fully_vaccinated_per_hundred']=vacc_pais['people_fully_vaccinated_per_hundred'].interpolate(method='cubicspline')
    vacc_pais['daily_vaccinations_per_million']= vacc_pais['daily_vaccinations_per_million'].interpolate(method='cubicspline')

    vacc_pais = vacc_pais.fillna(vacc_pais.median())
    
    vacc_pais["date"] = vacc_pais.index
    vacc_pais["Days"]=vacc_pais.date - vacc_pais.date.min()
    vacc_pais["Days"]=vacc_pais["Days"].dt.days
    
    population = vacc_pais["Population"][1]
    vacc_pais['percentage_vaccinated'] = (vacc_pais.people_fully_vaccinated/population)*100
    
    return vacc_pais



#--------procedimiento de plot -- ----- -- - - - - -  --  - - - - - -

##Procedimiento nº1
def plot_variables(data, variable1, variable2):
    sns.set_style("darkgrid")
    plt.figure(figsize=(16, 3))
    plt.subplot(121)
    plt.title(variable1)
    sns.lineplot(data=data[variable1])
    #sns.lineplot(data = vacc_Spain['daily_vaccinations'])
    plt.title(variable1,fontsize=15)
    plt.xlabel('Date',fontsize=15)
    plt.ylabel(variable1, fontsize=15)

    plt.subplot(122)
    plt.title(variable2)
    sns.lineplot(data=data[variable2])
    plt.title(variable2,fontsize=15)
    plt.xlabel('Date',fontsize=15)
    plt.ylabel(variable2, fontsize=15)
    plt.show()

    
#Procedimiento nº2     #media semanal y mensual
def weekly_monthly_avg(vacc_pais,variable1,variable2):
    vacc_weekly_avg = vacc_pais.resample('W').apply(np.mean)
    vacc_monthly_avg = vacc_pais.resample('M').apply(np.mean)
    
    plt.figure(figsize=(10,4))
    plt.subplot(221)
    plt.title('Vaccination week Avg', loc='center')
    plt.plot(vacc_weekly_avg[variable1], "-o", markersize=3, color='teal')
    plt.subplot(222)
    plt.title('Vaccination monthly Avg', loc='center')
    plt.plot(vacc_monthly_avg[variable1], "-o", markersize=3, color='teal')

    plt.subplot(223)
    plt.title('Vaccination week Avg', loc='center')
    plt.plot(vacc_weekly_avg[variable2], "-o", markersize=3, color='teal')
    plt.subplot(224)
    plt.title('Vaccination monthly Avg', loc='center')
    plt.plot(vacc_monthly_avg[variable2], "-o", markersize=3, color='teal')
    plt.show()
    
    
    
#Procedimento nº3
def rolling_mean(vacc_pais,variable1,variable2):
    rolling_mean_1 = vacc_pais[variable1].rolling(window=7, center=False).mean() #window of 7 (weekly avg) captures our data better 
    rolling_mean_2 = vacc_pais[variable2].rolling(window=7, center=False).mean() #window of 7 (weekly avg) captures our data better
    
    plt.figure(figsize=(16,3))
    plt.subplot(121)
    plt.plot(vacc_pais[variable1], color='teal', label = variable1)
    plt.plot(rolling_mean_1, 'red',label = 'media móvil')
    plt.title(variable1)
    plt.legend()

    plt.subplot(122)
    plt.plot(vacc_pais[variable2], color='teal', label = variable2)
    plt.plot(rolling_mean_2, 'red',label = 'media móvil')
    plt.title(variable2)
    plt.legend()

    

#Procedimento nº4, autocorrelación   
def autocorrelation(vacc_pais,variable1,variable2):
    lags = np.arange(2,100,3) #elegir desfase
    autocorrs_variable1 = [vacc_pais[variable1].autocorr(lag=lag) 
                       for lag in lags]


    plt.figure(figsize=(16, 3))
    plt.subplot(121)
    plt.stem(lags, autocorrs_variable1)
    plt.title(variable1)
    plt.xlabel("Lag", fontsize=12)
    plt.ylabel("Autocorrelation", fontsize=12)

    autocorrs_variable2 = [vacc_pais[variable2].autocorr(lag=lag) 
                       for lag in lags]
    plt.subplot(122)
    plt.stem(lags, autocorrs_variable2)
    plt.title('total_vaccinations')
    plt.xlabel("Lag", fontsize=12)
    plt.ylabel("Autocorrelation", fontsize=12)

    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
