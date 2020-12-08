import matplotlib as mpl
from matplotlib.dates import DayLocator, MonthLocator
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pandas as pd
import scipy.stats as stats
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

    return df


# vars

pd.options.display.max_rows = 500

cases_confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
                      '/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv '
cases_deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
                   '/csse_covid_19_time_series/time_series_covid19_deaths_US.csv '

selection_counties_list = ['Los Angeles']
# selection_counties_list = ['Alameda', 'Contra Costa', 'Marin', 'Napa', 'San Francisco', 'San Mateo', 'Santa Clara',
#                           'Solano', 'Sonoma']
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


# plot setup
mpl.style.use('Solarize_Light2')
mpl.rcParams['text.color'] = '#4f6066'
font_size = 10
plt.rcParams['font.size'] = font_size
custom_tick_color = '#d4cfbe'
fig = plt.figure(tight_layout=True, figsize=(10, 8))
gs = gridspec.GridSpec(2, 1)
ax = fig.add_subplot(gs[0, 0])
axs1 = fig.add_subplot(gs[1, 0])


# top fig: totals and averages normalized per 100k
print('Plotting totals and averages...')

ax_line = ax.plot(dates_cases, normed_cases_dataset['sum'], label='Total - Nine Bay Area Counties',
                  linewidth=1, color='#99958a')
ax.plot(dates_cases, normed_cases_dataset['avg_sum_7day'], label='Moving 7-Day Average',
        linewidth=2, color='#2AA198')
ax.plot(dates_cases, normed_cases_dataset['avg_sum_14day'], label='Moving 14-Day Average',
        linewidth=2, color='#D33682')
ax.plot(dates_cases, normed_cases_dataset['avg_sum_28day'], label='Moving 28-Day Average',
        linewidth=2, color='#6C71C4')

ax_leg = ax.legend(loc='upper left', title=r'$\bf{New}$' + ' ' +
                                           r'$\bf{Confirmed}$' + ' ' +
                                           r'$\bf{Cases}$' + ' ' +
                                           r'$\bf{Per}$' + ' ' +
                                           r'$\bf{100k}$')
ax_leg._legend_box.align = 'left'
ax.set_ylim(0, 120)
ax.set_xlim(dates_cases[131], right=max(dates_cases))
ax.xaxis.set_major_locator(MonthLocator(bymonth=range(6, 13)))
ax.xaxis.set_minor_locator(DayLocator(bymonthday=[15]))
ax.tick_params(which='both', color=custom_tick_color)


# bottom fig: calc curve fits for 28-day moving average, range defines number of curves (degrees)
print('Fitting curves for projections...')

# filter dates to only use last 28 days for curve regression
skip_days = len(dates_cases) - 28
normed_cases_dataset_skipped = normed_cases_dataset[skip_days:]
xdata = dates_cases[skip_days:]

# project out n days
projection_n = 28
xdata_plus = xdata
last_index = xdata_plus.last_valid_index()

last_day = xdata_plus.iat[-1]
xdata_last = pd.Series(last_day)

for i in range(1, projection_n):
    appended_day = pd.Series(data=last_day + pd.DateOffset(days=i), index=[last_index + i])
    xdata_plus = xdata_plus.append(appended_day)
    xdata_last = xdata_last.append(appended_day)

ydata = normed_cases_dataset_skipped['avg_sum_28day'].to_numpy()
xdata_index = xdata.index.to_numpy()

xdata_plus_index = xdata_plus.index.to_numpy()
xdata_last_index = xdata_plus_index[-1 - projection_n:-1]


# create regression curves and evaluate model fitness # https://stackoverflow.com/a/28336695
for i in range(1, 5):
    curve_params = np.polyfit(xdata_index, ydata, i, full=True)
    p = curve_params[0]
    curve = np.poly1d(p)
    y_model = curve(xdata_index)

    n = ydata.size
    k = p.size
    dof = n - k
    s_err = stats.sem(y_model)

    # https://pypi.org/project/RegscorePy/  BIC, lower is better
    resid = np.subtract(y_model, ydata)
    rss = np.sum(np.power(resid, 2))
    bic = round(n * np.log(rss/n) + k * np.log(n), 1)

    x2 = np.linspace(np.min(xdata_index), np.max(xdata_index), 100)
    y2 = curve(x2)
    t = stats.t.ppf(0.99, dof)
    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(xdata_index))**2 / np.sum((xdata_index - np.mean(xdata_index))**2))
    ci *= 1.2    # exaggerate
    ci_upper = np.poly1d(np.polyfit(x2, y2 + ci, i))
    ci_lower = np.poly1d(np.polyfit(x2, y2 - ci, i))

    axs1.plot(xdata_plus, curve(xdata_plus_index), linewidth=1,
              label=str(i) + '-Degree Fit, BIC = ' + str(bic))
    axs1.fill_between(xdata_plus, ci_upper(xdata_plus_index), ci_lower(xdata_plus_index), alpha=0.3)


# format lower plot
xlim_left = dates_cases[skip_days + 27]
ylim_bottom = 0

axs1_leg = axs1.legend(loc='upper left', title=r'$\bf{28}$' + '-' + r'$\bf{Day}$' + ' ' + r'$\bf{Projection}$')
axs1_leg._legend_box.align = 'left'

axs1.set_ylim(ylim_bottom, 450)
axs1.set_xlim(left=xlim_left, right=max(xdata_plus))
axs1.xaxis.set_major_locator(DayLocator(interval=7))
axs1.xaxis.set_minor_locator(DayLocator(interval=1))
axs1.tick_params(which='both', color=custom_tick_color)


# show figure
print('Displaying plot...')
plt.show()
print('Exiting...')
