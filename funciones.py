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
    - covid_vaccine_data: dataframe con todas las vacunas
    - pais: pais que queremos escoger para analizar
    
    return: dataframe preprocesado.
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


#Función 2, reorganización de los datos
def organize_data(to_forecast, window, horizon=1):
    
    shape = to_forecast.shape[:-1] + (to_forecast.shape[-1] - window + 1, window)
    strides = to_forecast.strides + (to_forecast.strides[-1],)
    X = np.lib.stride_tricks.as_strided(to_forecast,
                                        shape=shape,
                                        strides=strides)
    y = np.array([X[i+horizon][-1] for i in range(len(X)-horizon)])
    return X[:-horizon], y




def lr_series(time_series_people,time_series_total,lag,variable1,variable2):
    """
    input: dos series:
    - time_series_people, variable 1
    - time_series_total, variable 2
    - lag, el desfase, int
    - variable1, nombre de la variable a predecir ,str
    - variable2, nombre de la variable a predecir, str
    
    return: lista de listas con las prestaciones de los dos modelos de regresión: [[MAE_1,MAE2_1],[MAE_2,MAE2_2]]
    
    """
    
    lag = lag
    X_1, y_1 = organize_data(np.array(time_series_people), lag)
    X_2, y_2 = organize_data(np.array(time_series_total), lag)
    
    #variable1
    lr = LinearRegression()
    lr_fit_people = lr.fit(X_1, y_1)
    lr_prediction_people = lr_fit_people.predict(X_1)
    #variable2
    lr_fit_total = lr.fit(X_2, y_2)
    lr_prediction_total = lr_fit_total.predict(X_2)
    
    plt.figure(figsize=(17, 4))
    plt.subplot(121)
    plt.plot(time_series_people.values, '-o', color='teal')
    plt.plot(np.arange(lag, len(time_series_people)), lr_prediction_people, '-o', label='prediction', color='orange')
    plt.title('Linear regression:', variable1)
    plt.legend(fontsize=12);

    MAE_var1 = mean_absolute_error(time_series_people[lag:], lr_prediction_people)
    MAE2_var1 = mean_absolute_error(time_series_people[-90:], lr_prediction_people[-90:])

    print(variable1,': ')
    print('MAE = {0:.3f}'.format(MAE_var1))
    print('MAE2 = {0:.3f}'.format(MAE2_var1)) #for the last 90 days only

    print(' \n ')
    plt.subplot(122)
    plt.plot(time_series_total.values, '-o', color='teal')
    plt.plot(np.arange(lag, len(time_series_total)), lr_prediction_total, '-o', label='prediction', color='orange')
    plt.title('Linear regression model: total_vaccinations')
    plt.legend(fontsize=12);
    
    MAE_var2 = mean_absolute_error(time_series_total[lag:], lr_prediction_total)
    MAE2_var2 = mean_absolute_error(time_series_total[-90:], lr_prediction_total[-90:])

    print(variable2, ': ')
    print('MAE = {0:.3f}'.format(mean_absolute_error(time_series_total[lag:], lr_prediction_total)))
    print('MAE2 = {0:.3f}'.format(mean_absolute_error(time_series_total[-90:], lr_prediction_total[-90:]))) #for the last 90 days only
    
    return [[MAE_var1, MAE2_var1],[MAE_var2,MAE2_var2]]



#--------procedimiento de plot -- ----- -- - - - - -  --  - - - - - -

##Procedimiento nº1
def plot_variables(data, variable1, variable2):
    """
    - data = dataframe del país de interés, DataFrame, pandas
    - variable1 = variable que se quiere visualizar, str
    - variable2 = variable que se quiere visualizar, str
    
    """
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
    """
    - data = dataframe del país de interés, DataFrame, pandas
    - variable1 = variable que se quiere visualizar, str
    - variable2 = variable que se quiere visualizar, str
    
    """
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
    """
    - info: cálculo y visualización de media móvil
    
    data = dataframe del país de interés, DataFrame, pandas
    variable1 = variable que se quiere visualizar, str
    variable2 = variable que se quiere visualizar, str
    """
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
    
    
    
    
    
  
    
    
    
    
    
    
