# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:04:41 2023

@author: Gabriel
"""
import numpy as np
import pandas as pd
import datetime
import calendar
import matplotlib.pyplot as plt
from scipy.stats import beta
from pyDOE import lhs

def betaPDF(ghi):
    # Calculate the mean and variance of the data
    mean = np.mean(ghi)
    variance = np.var(ghi)

    # Calculate alpha and beta
    b = (1 - mean) * ((mean * (1 + mean) / variance) - 1)
    a = (mean * b) / (1 - mean)

    # Check if a or b is zero or close to zero
    if a <= 0 or b <= 0:
        raise ValueError("Invalid shape parameters. Cannot compute Beta PDF.")

    x = np.linspace(0, 1, 100)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = beta.pdf(x, a, b)

    # Latin Hypercube Sampling
    lhs_samples = lhs(n=1, samples=1000)
    samples = beta.ppf(lhs_samples, a, b).reshape(-1)

    return [x, y, samples]

# Read the CSV file
data_path = 'C:/Users/Luis Felipe Giraldo/OneDrive - Universidad de los andes/Doctorado/LAAS/Solar_Forecasting/data_complete/'
data = pd.read_csv(data_path + 'PHV.KIMO.PYRANO.CR100_1.MES1_1_2021_au_31_12_2021.csv', header=None, index_col=False, usecols=[1, 2])

# Calculate the mean every hour
data.rename(columns={2: 'irradiance'}, inplace=True)
data['irradiance'] = data['irradiance'] / 1000
hourly_mean = data['irradiance'].groupby(data.index // 60).mean()

def monthly_beta_samples(month_start, month_end):
    start_date = datetime.datetime(2021, 1, 1) + datetime.timedelta(days=(month_start - 1))

    hourly_mean_ghi = hourly_mean[(month_start - 1) * 24:month_end * 24].values.reshape(-1, 24)

    # Generate samples
    N = hourly_mean_ghi.shape[1]
    irradiance_samples = np.zeros((1000, 24))
    for i in range(N):
        [x, y, samples] = betaPDF(hourly_mean_ghi[:, i])
        irradiance_samples[:, i] = samples

    # Plot mean ghi
    plt.plot(np.mean(hourly_mean_ghi, axis=0), 'b')
    plt.plot(np.mean(irradiance_samples, axis=0), 'r--')
    plt.xlabel('Time [h]')
    plt.ylabel('Irradiance kW/m^2')
    month_name = start_date.strftime('%B')
    plt.title(f'Mean Solar Irradiance - {month_name}')
    plt.legend(['Historique', 'Samples'])
    plt.ylim((0, 0.8))
    plt.show()

    mean = np.mean(irradiance_samples, axis=0)
    variance = np.var(irradiance_samples, axis=0)

    return [irradiance_samples, mean, variance]

months = [(1, 31), (32, 59), (60, 90), (91, 120), (121, 151), (152, 181), (182, 212), (213, 243), (244, 273), (274, 304), (305, 335), (336, 365)]
irradiance_samples_months = []
for month_start, month_end in months:
    monthly_samples = monthly_beta_samples(month_start, month_end)
    irradiance_samples_months.append(monthly_samples)

n_samples = 20
for i, (samples, mean, _) in enumerate(irradiance_samples_months):
    month_name = calendar.month_name[i+1]
    # Plot samples
    for j in range(n_samples):
        plt.plot(samples[j])
    plt.xlabel('Time [h]')
    plt.ylabel('Irradiance kW/m^2')
    plt.title(f'Solar Irradiance Samples - {month_name}')
    plt.ylim((0, 0.8))
    plt.plot(mean, 'k--', linewidth=3, label='Mean')
    plt.legend()
    plt.show()