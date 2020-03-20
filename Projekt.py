import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# define function
def statistics(data, qvalue1, qvalue3):
    # funkcja liczy statystyki dla podanych danych i zwraca słownik
    stat_dict = {}
    stat_dict['minimum'] = np.amin(data)
    stat_dict['maximum'] = np.amax(data)
    stat_dict['range'] = np.ptp(data)
    stat_dict['quantile1'] = np.quantile(data, qvalue1)
    stat_dict['quantile3'] = np.quantile(data, qvalue3)
    stat_dict['median'] = np.median(data)
    stat_dict['mean'] = np.mean(data)
    stat_dict['std'] = np.std(data)
    stat_dict['cov'] = np.cov(data)
    return stat_dict

# data import with pandas
data = pd.read_csv("kurs-GBP_PLN.csv", delimiter = ';')
date = data.iloc[:,0]
y = data.iloc[:,1]
y = np.log(y)


# calculating statistics
stats = statistics(y, 0.25, 0.75)
print(stats)

# normal distribution
fig, axis = plt.subplots()    # funkcja subplots() zwraca dwa obiekty: klasy Figure(okno) i klasy Axis(osie)
y.plot.kde(color = "black")

# histogram
y.plot.hist( bins = 20, rwidth=0.9, density = True, color = "purple")
axis.grid(axis = 'y')
plt.title("Histogram")
plt.ylabel("GBP/PLN exchange rate")

# boxplot
fig2, axis2 = plt.subplots()
y.plot.box(color = 'brown')
axis2.grid(axis = 'y')
plt.title("Boxplot")

# timeseries
fig3, axis3 = plt.subplots()
plt.plot(date, y, color='green')
plt.xticks(np.linspace(0, len(date)-1, 10, endpoint=True), rotation=20)
axis3.grid(axis = 'y')
plt.title("Time series")
plt.ylabel("GBP/PLN exchange rate")
plt.show()

###### AR(p) model #####

from statsmodels.tsa.arima_model import ARMA

### Fit an AR(p) model into data ###

#Bayesian Information Criterion
BIC = np.zeros(12) # tworzy kontener (numpy array) zer o wymiarach 1x12
for p in range(12):
    model_bic = ARMA(y, order=(p, 0))
    fitted_bic = model_bic.fit()
    BIC[p] = fitted_bic.bic

plt.plot(range(1,12), BIC[1:12], marker='o', color='pink')
plt.xlabel('Order of AR Model')
plt.ylabel('Bayesian Information Criterion')
plt.show()

#Akaike Information Criterion
AIC = np.zeros(12) # tworzy kontener (numpy array) zer o wymiarach 1x12
for p in range(12):
    model_aic = ARMA(y, order=(p, 0))
    fitted_aic = model_aic.fit()
    AIC[p] = fitted_aic.aic

plt.plot(range(1,12), AIC[1:12], marker='o', color='green')
plt.xlabel('Order of AR Model')
plt.ylabel('Akaike Information Criterion')
plt.show()

#PACF
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(y, lags=24)
plt.show()

#Choosing a proper number of lags which is 1 (based on Akaike and Bayesian Information Criterion)
lags, = np.where(AIC == min(AIC[1:12]))
model = ARMA(y, order=(lags[0],0))
fitted = model.fit(method = 'mle') #Maximum likelihood
print(fitted.summary())


##### Residuals check #####
residuals = fitted.resid

#Normality + histogram
from scipy import stats
print(stats.shapiro(residuals))

residuals.plot.hist( bins = 20, rwidth=0.9, density = True, color = "pink")
axis.grid(axis = 'y')
plt.title("Histogram - residuals")
plt.ylabel("GBP/PLN exchange rate")

plot_pacf(residuals, lags=24)
plt.show()


#prediction

predicted = fitted.predict(start=0, end =100)

y = list(y)
predicted = list(predicted)

print(y)
predicted.insert(0, y[-1])
print(predicted)


predicted_x = []
for i in range(0, len(predicted)):
    predicted_x.append(len(y)+i)

plt.plot(y, color = "cyan")
plt.plot(predicted_x, predicted, color="purple")
plt.title("Forecast values")
plt.grid(axis='y')
plt.show()