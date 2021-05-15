## FUNCIONES DEL NOTEBOK DE SERIES TEMPORALES:
#------------------------------------------------------------------

#Importamos las librerías


import pandas as pd
from pandas import datetime as dt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import Statsmodels
from statsmodels.api import tsa

from statsmodels.tsa.arima_model import ARIMA

import statsmodels.api as sm
#VAR
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import acf

#plot sección 2
import inspect


## SECCIÓN UNO: MODELOS DE SERIES TEMPORALES

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
    plt.title('Linear regression:'+ variable1)
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
    plt.title('Linear regression model:'+ variable2)
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
    
    
#Procedimiento nº5, Modelo de series temporal:

def time_series_model(vacc_pais,time_series_people,time_series_total,optlag_people,optlag_total,model,days):
    
    people_fully_vaccinated = vacc_pais.people_fully_vaccinated
    people_fully_vaccinated_diff =people_fully_vaccinated.diff(2)
    
    total_vaccinations = vacc_pais.total_vaccinations
    total_vaccinations_diff =total_vaccinations.diff(2)
    
    #
    
    #AR, importante hacerlo ahora porque lo voy a utilizar para el ARIMA
    optlag_people = optlag_people
    ar_people = tsa.AR(time_series_people)
    ar_fit_people = ar_people.fit(maxlag=optlag_people)
    ar_forecast_people = ar_fit_people.predict(end=len(time_series_people)+(days-1))[-(days):] 
    #ar_forecast_people
    
    optlag_total = optlag_total
    ar_total = tsa.AR(time_series_total)
    ar_fit_total = ar_total.fit(maxlag=optlag_total)
    ar_forecast_total = ar_fit_total.predict(end=len(time_series_total)+(days-1))[-(days):] 
    #ar_forecast_total
    
    if model == 'AR':
        
        print('Pronóstico de la primera variable AR: ')
        print(ar_forecast_people)
        print('\n')
        print('Pronóstico de la segunda variable AR: ')
        print(ar_forecast_total)
        
        #que ploteé el gráfico del pronóstico
        
        #people
        plt.figure(figsize=(17, 4))
        plt.subplot(121)
        plt.plot(time_series_people, '-o', label="original data", color='teal')
        plt.plot(ar_forecast_people, '--o', label='prediction', color='orange')
        plt.title('Forecast AR, primera variable: people fully vaccinated')
        plt.legend(fontsize=12)

        #total_vaccinations
        plt.subplot(122)
        plt.plot(time_series_total, '-o', label="original data", color='teal')
        plt.plot(ar_forecast_total, '--o', label='prediction', color='orange')
        plt.title('Forecast AR, segunda variable: total vaccinations')
        plt.legend(fontsize=12)
    
    
    if model == 'ARMA':
        #implementar ARMA
    
        #7 out of sample prediction with ARMA
        #people fully vaccinated
        arma_people = tsa.ARMA(time_series_people, order=(2, 1)) 
        arma_people = arma_people.fit()
        arma_forecast_people = arma_people.predict(end=len(time_series_people)+(days-1))[-(days):]
        #arma_forecast_people

        #total vaccinations
        arma_total = tsa.ARMA(time_series_total, order=(2, 1)) 
        arma_total = arma_total.fit()
        arma_forecast_total = arma_total.predict(end=len(time_series_total)+(days-1))[-(days):]
        #arma_forecast_total
        
        print('Pronóstico de la primera variable ARMA: ')
        print(arma_forecast_people)
        print('\n')
        print('Pronóstico de la segunda variable ARMA: ')
        print(arma_forecast_total)
        
        #plot de ARMA
        
        #ARMA model's 7 out sample predicitons 
        plt.figure(figsize=(16, 4))
        plt.subplot(121)
        plt.plot(time_series_people, '-o', label="original data", color='teal')
        plt.plot(arma_forecast_people, '--o', label='prediction', color='orange')
        plt.legend(fontsize=12)
        plt.title('Forecast ARMA: people_fully_vaccinated')

        plt.subplot(122)
        plt.plot(time_series_total, '-o', label="original data", color='teal')
        plt.plot(arma_forecast_total, '--o', label='prediction', color='orange')
        plt.legend(fontsize=12)
        plt.title('Forecast ARMA: total vaccinations')

        plt.show()
    
    if model == 'ARIMA':
        
        #ARIMA: people fully vaccinated 
        model_people = ARIMA(time_series_people, order=(1,1,1))
        arima_fit_people = model_people.fit()
        arima_forecast_people= arima_fit_people.forecast(steps=days)[0]
        print('ARIMA: total vaccinations')
        print(arima_forecast_people)

        #ARIMA: totalvaccinations
        model_total = ARIMA(time_series_total, order=(1,1,1))
        arima_fit_total = model_total.fit()
        arima_forecast_total= arima_fit_total.forecast(steps=days)[0]
        print('ARIMA, total vaccinations:')
        print(arima_forecast_total)
        
        #people
        idx_people = ar_forecast_people.index.values
        forecast_people_fully_vaccinated = []
        lag = 7
        
        #Escogemos el model AR porque tiene un MAE menor que el modelo ARMA 
        
        for i, diff in enumerate(ar_forecast_people): 
            prev_value_people = people_fully_vaccinated[-(lag)+i:][0]
            forecast_people_fully_vaccinated.append(prev_value_people+diff)

        people_fully_vaccinated_forecast = pd.Series(forecast_people_fully_vaccinated, index=idx_people)
        print('Forecast ARIMA: \n')
        print(people_fully_vaccinated_forecast)

        #total_vaccinations
        idx_total = ar_forecast_total.index.values
        forecast_total = []

        for i, diff in enumerate(ar_forecast_total): #choosing AR as it produced lower MAE than ARMA model
            prev_value_total = total_vaccinations[-(lag)+i:][0]
            forecast_total.append(prev_value_total+diff)

        total_vaccinations_forecast = pd.Series(forecast_total, index=idx_total)
        print('\n')
        print('Forecast ARIMA: \n')
        print(total_vaccinations_forecast)
        
        hist_values_people = vacc_pais['people_fully_vaccinated'].append(people_fully_vaccinated_forecast)
        hist_values_total = vacc_pais['total_vaccinations'].append(total_vaccinations_forecast)
        
        #plot de ARIMA

        plt.figure(figsize=(17,4))
        plt.subplot(121)
        plt.plot(hist_values_people, '-o', color='teal', alpha=0.5)
        plt.plot(people_fully_vaccinated_forecast, '--o', label='prediction', color='orange')
        plt.legend()
        plt.title('Forecast ARIMA prediction: people_fully_vaccinated')
        plt.xlim('2021-02-25','2021-05-25')

        plt.subplot(122)
        plt.plot(hist_values_total, '-o', color='teal', alpha=0.5)
        plt.plot(total_vaccinations_forecast, '--o', label='prediction', color='orange')
        plt.legend()
        plt.title('Forecast ARIMA prediction: total_vaccinations')
        plt.xlim('2021-02-25','2021-05-25')
        
    if model == 'VAR':
       
        return ar_forecast_people
        
      
        
         
    
#---


  

#Función, test de Granger
def grangers_causation_matrix(data, variables, maxlag,test='ssr_chi2test', verbose=False):
    """Compruebe la causalidad de Granger de todas las combinaciones posibles de las series temporales. Las filas son la 
         variable de respuesta, las columnas son los predictores. Los valores de la tabla son los valores P. Los valores P 
         menores que el nivel de significación (0,05), implican la hipótesis nula de que los coeficientes de los valores 
        pasados correspondientes son cero, es decir, que la X no causa la Y puede ser rechazada.
        data      : pandas dataframe containing the time series variables
        variables : list containing names of the time series variables. """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


#Procedimiento, test de Cointegración:    
def cointegration_test(df, alpha=0.05): 
    
    """Perform Johanson's Cointegration Test and Report Summary"""
    
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt) 
        

#Función para comprobar estocionariedad:       
def adfuller_test(series, signif=0.05, name='', verbose=False):
    
    """Perform ADFuller to test for Stationarity of given series and print report"""
    
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")   
        
#referencia: https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
    

#invertir la diferenciación
#Función
def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc
    
#Funciones accuracy
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})




#VAR

def VAR_model_pais(vacc_Spain,variables,
                  time_series_people, time_series_total,optlag_people,optlag_total,pais,model='VAR',days=7):
    
    var_Spain = vacc_Spain[variables]
    
    # Splitting the dataset into training and test data.
    nobs = 7
    df_train, df_test = var_Spain[0:-nobs], var_Spain[-nobs:]
    
    
    # First Differencing
    var_Spain_differenced = var_Spain.diff().dropna()
    #second differencing
    var_Spain_differenced = var_Spain_differenced.diff().dropna()
    
    ar_forecast_people = time_series_model(var_Spain,time_series_people,
                                         time_series_total,optlag_people,optlag_total,model,days=7)
    
    idx_20 = ar_forecast_people.index.values

    forecast_input = var_Spain.values
    
    #MODELO
    model = VAR(var_Spain_differenced)
    maxlags=12
    x = model.select_order(maxlags) ## hacer una variable de entrada el maxlags
    model_fitted = model.fit(maxlags)
    lag_order = model_fitted.k_ar
    forecast_input = var_Spain_differenced.values[-lag_order:]

    fc = model_fitted.forecast(y=forecast_input, steps=7)
    df_forecast = pd.DataFrame(fc, index=idx_20, columns=var_Spain.columns + '_2d')
    
    df_results = invert_transformation(var_Spain, df_forecast, second_diff=True) 
    
    df_inverted = df_results.loc[:, ['total_vaccinations_forecast', 'people_vaccinated_forecast',
                   'people_fully_vaccinated_forecast', 'daily_vaccinations_forecast']]
    
    
    
    fig, axes = plt.subplots(nrows=int(len(var_Spain.columns)/2), ncols=2, dpi=150, figsize=(7,7))
    for i, (col,ax) in enumerate(zip(var_Spain.columns, axes.flatten())):
        df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
        df_test[col][-nobs:].plot(legend=True, ax=ax);
        ax.set_title(col + ' ' + pais +":Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

    plt.tight_layout();
    

    
###

#Función de varios países

def VAR_paises(covid_vaccine_data,paises):
    """
    covid_vaccine_data: DataFrame con los datos de todos los países
    paises: lista de países de los que queremos saber su pronóstico, lst
    """
    for i in paises:
    
        vacc_pais = f.preprocesado_pais(covid_vaccine_data,i)
        people_fully_vaccinated = vacc_pais.people_fully_vaccinated
        people_fully_vaccinated_diff =people_fully_vaccinated.diff(2)

        total_vaccinations = vacc_pais.total_vaccinations
        total_vaccinations_diff =total_vaccinations.diff(2)

        time_series_people = people_fully_vaccinated_diff
        time_series_total = total_vaccinations_diff

        #time_series, imputamos los NaN, porque quedan los del lag del principio
        time_series_people = time_series_people.fillna(time_series_people.median())
        time_series_total = time_series_total.fillna(time_series_total.median())

        #planteamos el modelo AR: people
        ar_people = tsa.AR(time_series_people)
        #calculamos cual es el lag o el desfase adecuado 
        optlag_people = ar_people.select_order(20, ic='aic') 
        #planteamos el modelo AR: total, 
        #porque vamos a utilizar el index, es decir las fechas, para el modelo VAR
        ar_total = tsa.AR(time_series_total)
        #calculamos cual es el lag o el desfase adecuado 
        optlag_total = ar_total.select_order(20, ic='aic') 


        #print(i)
        f.VAR_model_pais(vacc_pais,variables,time_series_people,
                   time_series_total,optlag_people,optlag_total,pais=i,model='VAR',days=7)



## SECCIÓN DOS: AJUSTE DE CURVA DE TENDENCIA

#funciones previas;
import inspect
def select_df(df, **kwargs):
    attrs = df.attrs
    for k, vs in kwargs.items():
        if vs is None:
            df = df[df.__getitem__(k).isna()]
        elif not isinstance(vs, list):
            df = df[df.__getitem__(k) == vs]
        else:
            df = df[df.__getitem__(k).isin(vs)]
    df.attrs = attrs
    return df


def _augment_df(df, fn, name=None, register=None):
    name = fn.__name__ if name is None else name
    params = list(inspect.signature(fn).parameters.keys())
#     fixed = {p: df.attrs["uniq"][p] for p in params if p not in df.columns}
#     params = [p for p in params if p not in fixed
#     if len(fixed) > 0:
#         fn = functools.partial(fn, **fixed)

    def wrapper(row):
        kwargs = {k: row.get(k) for k in params}
        return fn(**kwargs)

    df[name] = df.apply(wrapper, axis=1)

    if register:
        if not register in df.attrs:
            df.attrs[register] = []
        if name not in df.attrs[register]:
            df.attrs[register].append(name)


def augment_df(df, *fns, register=None):
    for f in fns:
        _augment_df(df, f, register=register)
        

#####----------------------

#curvas de ajuste

from scipy.optimize import curve_fit

def lineal(x, a, b):
    return a*x +b 

def powerlaw(x, a, b, c):
    return c*x**a + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def exp(x, a, b, c):
    return a**(x-c)+b

def logistic(x, a, b, c, d):
    return a/(1+np.exp(b*(x-c))) + d










