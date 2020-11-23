import matplotlib.pyplot as plt
import pandas as pd
import urllib.request as urll


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

    # extract dates
    df_dates = df.pop('Date')
    
    # split by column
    dfs = [df_dates]
    for col in df.columns:
        dfs.append(df[col])

    return dfs


def extract_populations(df, filter_col, pop_col, selection_counties_list):
    df2 = pd.DataFrame({'County': df[filter_col], 'Pop': df[pop_col]})
    print(df2)
    # transpose
    df2.reset_index(drop=True, inplace=True)
    df2 = df2.transpose()
    # print(df2)
    df2.reset_index(drop=False, inplace=True)
    df2 = df2[1:]
    df2.reset_index(drop=True, inplace=True)
    df2.pop('index')
    # print(df2)
    df2.columns = selection_counties_list
    # print(df2)

    # df_og = df
    # print(df_og)
    # df = pd.DataFrame(data=[df[filter_col], cases_deaths[pop_col]])
    # df_og = df
    # # print(df)
    # df.columns = df.iloc[0]
    # # print(df)
    # df = df[1:]
    # # print(df)
    # df.reset_index(drop=True, inplace=True)
    # print(df.columns)

    return(df2)

# vars

cases_confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
cases_deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'

selection_counties_list = ['Alameda', 'Contra Costa', 'Marin', 'Napa', 'San Francisco', 'San Mateo', 'Santa Clara', 'Solano', 'Sonoma']
selection_counties = pd.Series(selection_counties_list)
selection_cols_to_drop = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Province_State', 'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population']
filter_col = 'Admin2'

######

## scrape data
# urll.urlretrieve(cases_confirmed_url, 'cases_confirmed.csv')
# urll.urlretrieve(cases_deaths_url, 'cases_deaths.csv')


# extract county population totals
popu_counties = selection_counties_list
################### fix popu_counties['Population'].pop()
popu_counties = pd.Series(popu_counties)
cases_deaths_og = filter_csv('cases_deaths.csv', filter_col, popu_counties)
county_populations = extract_populations(cases_deaths_og, filter_col, 'Population', popu_counties)
print(county_populations)


# filter by county
cases_confirmed = filter_csv('cases_confirmed.csv', filter_col, selection_counties)
cases_deaths = filter_csv('cases_deaths.csv', filter_col, selection_counties)


# cleanup tables and convert to array of counties, index 0 is array of dates
cases_confirmed = preprocess_and_split(cases_confirmed, selection_counties_list, selection_cols_to_drop)
cases_deaths = preprocess_and_split(cases_deaths, selection_counties_list, selection_cols_to_drop)
print(cases_confirmed)
print(cases_deaths)





# cases_confirmed_percapita = cases_confirmed.div(county_pops, axis='columns')
# cases_confirmed_percapita = cases_confirmed_percapita.join(cases_dates)
# # print(cases_confirmed_percapita)


# for county in selection_counties_list:
#     plt.plot(cases_confirmed_percapita['Date'], cases_confirmed_percapita[county])

# plt.show()
