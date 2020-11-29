import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
import urllib.request as urll
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


def filter_csv(filepath, filter_col, filter_list):
    csv = pd.read_csv(filepath, sep=',', quotechar='\"', warn_bad_lines=True)
    return csv.loc[csv[filter_col].isin(filter_list)]


def preprocess_and_split(df, selection_counties_list, selection_cols_to_drop):
    # drop unwanted columns
    for col in selection_cols_to_drop:
        if col in df: df.pop(col)
    
    # transpose
    df.reset_index(drop=True, inplace=True)
    df = df.transpose()
    df.reset_index(drop=False, inplace=True)
    df = df[1:]
    df.reset_index(drop=True, inplace=True)
    df.columns = ['Date'] + selection_counties_list
    dates = df.pop('Date')

    return df, dates


def extract_populations(df, filter_col, pop_col, selection_counties_list):
    # filter
    df2 = pd.DataFrame({'County': df[filter_col], 'Pop': df[pop_col]})
    
    # transpose
    df2.reset_index(drop=True, inplace=True)
    df2 = df2.transpose()
    df2.reset_index(drop=False, inplace=True)
    df2 = df2[1:]
    df2.reset_index(drop=True, inplace=True)
    df2.pop('index')
    df2.columns = selection_counties_list

    return df2.to_dict('records')


def totals_to_deltas(df):
    df = df.diff(periods=1, axis='index')
    df = df.replace({None: 0})
    # df = df.where(df > 0, 0)  # don't do this, it gives bad totals

    return df


# vars

pd.options.display.max_rows = 500

cases_confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
cases_deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'

selection_counties_list = ['Alameda', 'Contra Costa', 'Marin', 'Napa', 'San Francisco', 'San Mateo', 'Santa Clara', 'Solano', 'Sonoma']
selection_counties = pd.Series(selection_counties_list)
selection_cols_to_drop = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Province_State', 'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population']
filter_col = 'Admin2'


# main

# # scrape data
urll.urlretrieve(cases_confirmed_url, 'cases_confirmed.csv')
urll.urlretrieve(cases_deaths_url, 'cases_deaths.csv')


# extract county population totals
popu_counties = selection_counties_list
popu_counties = pd.Series(popu_counties)
cases_deaths_og = filter_csv('cases_deaths.csv', filter_col, popu_counties)
county_populations = extract_populations(cases_deaths_og, filter_col, 'Population', popu_counties)
county_populations = county_populations[0]
# print(county_populations['Alameda'])


# filter by county
cases_confirmed = filter_csv('cases_confirmed.csv', filter_col, selection_counties)
cases_deaths = filter_csv('cases_deaths.csv', filter_col, selection_counties)


# cleanup tables and convert to array of counties data as pd.Series; index 0 is array of dates
cases_confirmed, dates1 = preprocess_and_split(cases_confirmed, selection_counties_list, selection_cols_to_drop)
cases_deaths, dates2 = preprocess_and_split(cases_deaths, selection_counties_list, selection_cols_to_drop)
# print(cases_confirmed)
# print(dates1)
# print(cases_deaths)


# calc daily cases from totals
cases_confirmed_daily = totals_to_deltas(cases_confirmed)
cases_deaths_daily = totals_to_deltas(cases_deaths)

# print(cases_confirmed_daily)
# print(cases_deaths_daily)
# print(cases_deaths_daily.sum(axis=0))

# cases and deaths per 100k, normalized by county population
cases_confirmed_county = cases_confirmed_daily
for county in cases_confirmed_county:
    co_pop = county_populations[county]
    cases_confirmed_county[county] = cases_confirmed_county[county].div(co_pop) * 100000.0
# print(cases_confirmed_county)
cases_deaths_county = cases_deaths_daily
for county in cases_deaths_county:
    co_pop = county_populations[county]
    cases_deaths_county[county] = cases_deaths_county[county].div(co_pop) * 100000.0
# print(cases_deaths_county)

# calc overall normalized cases per 100k and moving 7-day average
moving_7day_avg = cases_confirmed_daily
moving_7day_avg['sum'] = moving_7day_avg.sum(axis=1)
moving_7day_avg['sum_7day'] = moving_7day_avg['sum'].rolling(window=7).sum()
moving_7day_avg['avg_sum_7day'] = moving_7day_avg['sum_7day'].div(7)
moving_7day_avg['sum_14day'] = moving_7day_avg['sum'].rolling(window=14).sum()
moving_7day_avg['avg_sum_14day'] = moving_7day_avg['sum_14day'].div(14)
moving_7day_avg['sum_28day'] = moving_7day_avg['sum'].rolling(window=28).sum()
moving_7day_avg['avg_sum_28day'] = moving_7day_avg['sum_28day'].div(28)
moving_7day_avg = moving_7day_avg.replace({None: 0})
# print(moving_7day_avg)


# filter dates to start 5/25 (Memorial Day)
skip_days = 120
moving_7day_avg = moving_7day_avg[skip_days:]
xdata = dates1.index.to_numpy()
xdata = xdata[skip_days:]
# print(moving_7day_avg)
# print(xdata)


# calc curve fit for 28 day
ydata = moving_7day_avg['avg_sum_28day'].to_numpy()

curve_params1 = np.polyfit(xdata, ydata, 4, full=True)
r_squared1 = curve_params1[1]
curve_p1 = np.poly1d(curve_params1[0])

curve_params2 = np.polyfit(xdata, ydata, 5, full=True)
r_squared2 = curve_params2[1]
curve_p2 = np.poly1d(curve_params2[0])


# # project out x days
# last = xdata[-1]
# for i in range(0, 14):
#     xdata = np.append(xdata, last + i)
#     i += 1
# # print(xdata)


# plot it!
fig, axs = plt.subplots(2, 1, figsize=(18, 9))   # one over one layout

# axs[0].plot(dates1, moving_7day_avg['sum'], label='Regional Total')
axs[0].plot(xdata, moving_7day_avg['avg_sum_7day'], label='Moving 7-Day Average')
axs[0].plot(xdata, moving_7day_avg['avg_sum_14day'], label='Moving 14-Day Average')
axs[0].plot(xdata, moving_7day_avg['avg_sum_28day'], label='Moving 28-Day Average')
axs[0].plot(xdata, curve_p1(xdata), label='Curve fit 4deg')
axs[0].plot(xdata, curve_p2(xdata), label='Curve fit 5deg')


axs[0].set_ylim(0)
axs[0].set_xlim(45)
axs[0].legend(loc='upper left', title='New Cases Per 100k Pop.')
axs[0].xaxis.set_major_locator(MultipleLocator(28))
axs[0].xaxis.set_minor_locator(AutoMinorLocator())
axs[0].grid()


for county in selection_counties_list:
    axs[1].plot(dates1, cases_confirmed_daily[county], label=county)

axs[1].set_ylim(0)
# xmax = axs[1].get_xlim()[1]
# axs[1].set_xlim(xmax - 75, xmax - 15)
axs[1].set_xlim(45)
axs[1].legend(loc='upper left', title='Recent Cases Per 100k Pop. by County')
axs[1].xaxis.set_major_locator(MultipleLocator(28))
axs[1].xaxis.set_minor_locator(AutoMinorLocator())
axs[1].grid()

plt.show()
