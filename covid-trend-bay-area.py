import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request as urll
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


def filter_csv(filepath, filter_col, filter_list):
    csv = pd.read_csv(filepath, sep=',', quotechar='\"', warn_bad_lines=True)
    return csv.loc[csv[filter_col].isin(filter_list)]


def preprocess_and_split(df, selection_counties_list, selection_cols_to_drop):
    # drop unwanted columns
    for col in selection_cols_to_drop:
        if col in df:
            df.pop(col)

    # transpose
    df.reset_index(drop=True, inplace=True)
    df = df.transpose()
    df.reset_index(drop=False, inplace=True)
    df = df[1:]
    df.reset_index(drop=True, inplace=True)

    df.columns = ['Date'] + selection_counties_list

    dates = df.pop('Date')
    dates = pd.to_datetime(dates, errors='raise', infer_datetime_format=True)

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
# print('Scraping data from https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/ ...')
# urll.urlretrieve(cases_confirmed_url, 'cases_confirmed.csv')
# urll.urlretrieve(cases_deaths_url, 'cases_deaths.csv')


print('Preprocessing...')

# extract county population totals
popu_counties = selection_counties_list
popu_counties = pd.Series(popu_counties)
cases_deaths_og = filter_csv('cases_deaths.csv', filter_col, popu_counties)
county_populations = extract_populations(cases_deaths_og, filter_col, 'Population', popu_counties)
county_populations = county_populations[0]


# filter by county
cases_confirmed = filter_csv('cases_confirmed.csv', filter_col, selection_counties)
cases_deaths = filter_csv('cases_deaths.csv', filter_col, selection_counties)


# cleanup tables and convert to array of counties data as pd.Series; index 0 is array of dates
cases_confirmed, dates_cases = preprocess_and_split(cases_confirmed, selection_counties_list, selection_cols_to_drop)
cases_deaths, dades_deaths = preprocess_and_split(cases_deaths, selection_counties_list, selection_cols_to_drop)


# calc daily cases from totals
cases_confirmed_daily = totals_to_deltas(cases_confirmed)
cases_deaths_daily = totals_to_deltas(cases_deaths)


# cases and deaths per 100k, normalized by county population
cases_confirmed_county = cases_confirmed_daily
for county in cases_confirmed_county:
    co_pop = county_populations[county]
    cases_confirmed_county[county] = cases_confirmed_county[county].div(co_pop) * 100000.0

cases_deaths_county = cases_deaths_daily

for county in cases_deaths_county:
    co_pop = county_populations[county]
    cases_deaths_county[county] = cases_deaths_county[county].div(co_pop) * 100000.0


# calc overall normalized cases per 100k and moving 7-day average
normed_cases_dataset = cases_confirmed_daily
normed_cases_dataset['sum'] = normed_cases_dataset.sum(axis=1)
normed_cases_dataset['sum_7day'] = normed_cases_dataset['sum'].rolling(window=7).sum()
normed_cases_dataset['avg_sum_7day'] = normed_cases_dataset['sum_7day'].div(7)
normed_cases_dataset['sum_14day'] = normed_cases_dataset['sum'].rolling(window=14).sum()
normed_cases_dataset['avg_sum_14day'] = normed_cases_dataset['sum_14day'].div(14)
normed_cases_dataset['sum_28day'] = normed_cases_dataset['sum'].rolling(window=28).sum()
normed_cases_dataset['avg_sum_28day'] = normed_cases_dataset['sum_28day'].div(28)
normed_cases_dataset = normed_cases_dataset.replace({None: 0})
# print(normed_cases_dataset)


# filter dates to start 5/25 (Memorial Day)
skip_days = 124
normed_cases_dataset = normed_cases_dataset[skip_days:]
xdata = dates_cases[skip_days:]


# project out 14 days
xdata_plus14 = xdata
last_index = xdata_plus14.last_valid_index()
last_day = xdata_plus14.iat[-1]

for i in range(1, 15):
    appended_day = pd.Series(data=last_day + pd.DateOffset(days = i), index=[last_index + i])
    xdata_plus14 = xdata_plus14.append(appended_day)


# plot setup, one over one layout
fig, axs = plt.subplots(2, 1, figsize=(18, 9))

# calc curve fits for 28-day moving average, range defines number of curves (degrees)
print('Fitting curves...')

ydata = normed_cases_dataset['avg_sum_28day'].to_numpy()
xdata_index = xdata.index.to_numpy()
xdata_plus14_index = xdata_plus14.index.to_numpy()


for i in range(5, 11):
    curve_params = np.polyfit(xdata_index, ydata, i, full=True)
    curve = np.poly1d(curve_params[0])
    r_squared = round(float(curve_params[1]), 1)
    axs[0].plot(xdata_plus14, curve(xdata_plus14_index), label = str(i) + '-deg. fit, R = ' + str(r_squared))

axs[0].plot(xdata, normed_cases_dataset['avg_sum_28day'], label='Moving 28-Day Average')
axs[0].scatter(last_day, normed_cases_dataset['avg_sum_28day'][last_index])

axs[0].legend(loc='upper left', title='New Cases Per 100k Pop.')
axs[0].set_ylim(0, 300)
axs[0].set_xlim(left=min(xdata_plus14), right=max(xdata_plus14))
axs[0].xaxis.set_major_locator(MultipleLocator(28))
axs[0].xaxis.set_minor_locator(AutoMinorLocator())
axs[0].grid()



# for county in selection_counties_list:
#     axs[1].plot(dates_cases, cases_confirmed_daily[county], label=county)


# # axs[0].plot(dates_cases, normed_cases_dataset['sum'], label='Regional Total')
# axs[0].plot(xdata, normed_cases_dataset['avg_sum_7day'], label='Moving 7-Day Average')
# axs[0].plot(xdata, normed_cases_dataset['avg_sum_14day'], label='Moving 14-Day Average')


# axs[1].set_ylim(0)
# # xmax = axs[1].get_xlim()[1]
# # axs[1].set_xlim(xmax - 75, xmax - 15)
# axs[1].set_xlim(45)
# axs[1].legend(loc='upper left', title='Recent Cases Per 100k Pop. by County')
# axs[1].xaxis.set_major_locator(MultipleLocator(28))
# axs[1].xaxis.set_minor_locator(AutoMinorLocator())
# axs[1].grid()

print('Plotting...')
plt.show()
print('Exiting...')
