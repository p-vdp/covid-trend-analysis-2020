import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.style
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
import pandas as pd
import urllib.request as urll
from sys import argv


def filter_csv(filepath, filter_column, filter_list):
    csv = pd.read_csv(filepath, sep=',', quotechar='\"', warn_bad_lines=True)
    return csv.loc[csv[filter_column].isin(filter_list)]


def preprocess_and_split(df, selection, selection_cols_drop):
    # drop unwanted columns
    for col in selection_cols_drop:
        if col in df:
            df.pop(col)

    # transpose
    df.reset_index(drop=True, inplace=True)
    df = df.transpose()
    df.reset_index(drop=False, inplace=True)
    df = df[1:]
    df.reset_index(drop=True, inplace=True)

    df.columns = ['Date'] + selection

    dates = df.pop('Date')
    dates = pd.to_datetime(dates, errors='raise', infer_datetime_format=True)

    return df, dates


def extract_populations(df, co_filter, pop_col, selection):
    # filter
    df2 = pd.DataFrame({'County': df[co_filter], 'Pop': df[pop_col]})

    # transpose
    df2.reset_index(drop=True, inplace=True)
    df2 = df2.transpose()
    df2.reset_index(drop=False, inplace=True)
    df2 = df2[1:]
    df2.reset_index(drop=True, inplace=True)
    df2.pop('index')
    df2.columns = selection

    return df2.to_dict('records')


def totals_to_deltas(df):
    df = df.diff(periods=1, axis='index')
    df = df.replace({None: 0})
    # df = df.where(df > 0, 0)  # don't do this, it gives bad totals

    return df


# vars

pd.options.display.max_rows = 500

cases_confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
                      '/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv '
cases_deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
                   '/csse_covid_19_time_series/time_series_covid19_deaths_US.csv '

selection_counties_list = ['Alameda', 'Contra Costa', 'Marin', 'Napa', 'San Francisco', 'San Mateo', 'Santa Clara',
                           'Solano', 'Sonoma']
selection_counties = pd.Series(selection_counties_list)
selection_cols_to_drop = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Province_State', 'Country_Region', 'Lat', 'Long_',
                          'Combined_Key', 'Population']
filter_col = 'Admin2'

scrape_flag = False
if len(argv) > 1 and argv[1] == '-d':
    scrape_flag = True

# main

# scrape data
if scrape_flag:
    print('Scraping data from https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data'
          '/csse_covid_19_time_series/ ...')
    urll.urlretrieve(cases_confirmed_url, 'cases_confirmed.csv')
    urll.urlretrieve(cases_deaths_url, 'cases_deaths.csv')

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


# plot setup, one over two layout
mpl.style.use('Solarize_Light2')
mpl.rcParams['text.color'] = '#4f6066'
custom_tick_color = '#d4cfbe'
fig = plt.figure(tight_layout=True, figsize=(18, 9))
gs = gridspec.GridSpec(2, 2)
ax = fig.add_subplot(gs[0, :])
axs1 = fig.add_subplot(gs[1, 0])
axs2 = fig.add_subplot(gs[1, 1])

# top fig: totals and averages normalized per 100k
print('Plotting totals and averages...')

ax_line = ax.plot(dates_cases, normed_cases_dataset['sum'], label='Total - All Bay Area Counties', linewidth=1.5)
ax.plot(dates_cases, normed_cases_dataset['avg_sum_7day'], label='Moving 7-Day Average', linewidth=3.5)
ax.plot(dates_cases, normed_cases_dataset['avg_sum_14day'], label='Moving 14-Day Average', linewidth=3.5)
ax.plot(dates_cases, normed_cases_dataset['avg_sum_28day'], label='Moving 28-Day Average', linewidth=3.5)
# ax.scatter(dates_cases.last_valid_index(), normed_cases_dataset.last_valid_index())

ax_leg = ax.legend(loc='upper left', title='New Cases Per 100k')
ax_leg._legend_box.align = 'left'
ax.set_ylim(0, 350)
ax.set_xlim(dates_cases[39], right=max(dates_cases))
ax.xaxis.set_major_locator(MultipleLocator(28))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', color=custom_tick_color)
# ax.grid()


# subfig 1: calc curve fits for 28-day moving average, range defines number of curves (degrees)
print('Fitting curves for 14-day projections...')


# filter dates to start 5/25 (Memorial Day)
skip_days = 124
normed_cases_dataset_skipped = normed_cases_dataset[skip_days:]
xdata = dates_cases[skip_days:]

# project out 14 days
xdata_plus14 = xdata
last_index = xdata_plus14.last_valid_index()

last_day = xdata_plus14.iat[-1]
xdata_last14 = pd.Series(last_day)

for i in range(1, 15):
    appended_day = pd.Series(data=last_day + pd.DateOffset(days=i), index=[last_index + i])
    xdata_plus14 = xdata_plus14.append(appended_day)
    xdata_last14 = xdata_last14.append(appended_day)


ydata = normed_cases_dataset_skipped['avg_sum_28day'].to_numpy()
xdata_index = xdata.index.to_numpy()

xdata_last14_index = xdata_plus14.index.to_numpy()
xdata_last14_index = xdata_last14_index[-16:-1]

for i in range(4, 11):
    curve_params = np.polyfit(xdata_index, ydata, i, full=True)
    curve = np.poly1d(curve_params[0])
    r_squared = round(float(curve_params[1]), 1)
    axs1.plot(xdata_last14, curve(xdata_last14_index), label=str(i) + '-deg. fit, R = ' + str(r_squared))

axs1_leg = axs1.legend(loc='upper left', title='14-day Projection (Curve Fit to 28-day Avg.)')
axs1_leg._legend_box.align = 'left'
axs1.set_ylim(50, 400)
axs1.set_xlim(left=last_day, right=max(xdata_plus14))
axs1.set_xticks([xdata_last14.iat[1], xdata_last14.iat[7], xdata_last14.iat[14]])
axs1.tick_params(which='both', color=custom_tick_color)
# axs1.xaxis.set_minor_locator(AutoMinorLocator(7))
# axs1.grid()


print('Plotting counties...')
for county in selection_counties_list:
    axs2.plot(dates_cases, normed_cases_dataset[county], label=county, linewidth=1.0)

axs2.set_ylim(0)
axs2.set_xlim(dates_cases[39], right=max(dates_cases))
axs2.tick_params(which='both', color=custom_tick_color)
axs2.xaxis.set_major_locator(MultipleLocator(48))
axs2.xaxis.set_minor_locator(AutoMinorLocator())
axs2_leg = axs2.legend(loc='upper left', title='New Cases Per 100k by County')
axs2_leg._legend_box.align = 'left'

# axs1.set_ylim(0)
# # xmax = axs1.get_xlim()[1]
# # axs1.set_xlim(xmax - 75, xmax - 15)
# axs1.set_xlim(45)
# axs1.legend(loc='upper left', title='Recent Cases Per 100k Pop. by County')
# axs1.xaxis.set_major_locator(MultipleLocator(28))
# axs1.xaxis.set_minor_locator(AutoMinorLocator())
# axs1.grid()

print('Displaying plot...')
plt.show()
print('Exiting...')
