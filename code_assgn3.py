import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from scipy.optimize import curve_fit

def read_data(file):
    """
    This function returns two dataframes
    data_year = year as columns
    data_country = country as columns
    Parameters
    ----------
    file : String
        File name of the string to be read.

    Returns
    -------
    Dataframe
        Returns dataframe

    """
    data = pd.read_excel(file, header=None)
    data = data.iloc[4:]
    var = data.rename(columns=data.iloc[0]).drop(data.index[0])
    list_col = ['Country Code', 'Indicator Name', 'Indicator Code']
    var = var.drop(list_col, axis=1)
    data_year = var.set_index("Country Name")
    
    
    data_country = data_year.transpose()
    data_year.index.name = None
    data_country.index.name = None
    return data_year.fillna(0), data_country.fillna(0)

    #return data_year.fillna(0)


def filter_dataframes(data_frame):
    """
    This function filters the dataframe that is being passed based on country
    names present in the list countries

    Parameters
    ----------
    data_frame : DataFrame
        DataFrame to be filtered based on countries.

    Returns
    -------
    data_frame : DataFrame
        Filtered DataFrame.

    """
    data_frame = data_frame[data_frame.index.isin(countries)]
    return data_frame


def filter_year_dataframe(data_frame):
    """
    This function filters the dataframe that is being passed based on years
    present in the list years

    Parameters
    ----------
    data_frame : DataFrame
        DataFrame to be filtered based on year.

    Returns
    -------
    data_frame : DataFrame
        Filtered DataFrame.


    """
    data_frame = data_frame[years]
    return data_frame


def norm(array):

    min_val = np.min(array)
    max_val = np.max(array)

    scaled = (array-min_val) / (max_val-min_val)

    return scaled


def norm_df(df, first=0, last=None):

    # iterate over all numerical columns
    for col in df.columns[first:last]:     # excluding the first column
        df[col] = norm(df[col])
    return df


def heat_corr(df, size=5):
    """Function creates heatmap of correlation matrix for each pair of columns 
    in the dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()


def func(x,a,b,c):
    return a * np.exp(-(x-b)**2 / c)


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper

'''adding an exponential function'''
def expoFunc(x,a,b):
    return a**(x+b)


countries = ["United States", "France", "China", "India", "Germany",
             "United Kingdom",
             "Japan", "Italy"]

# Range of years
years = range(2000, 2015, 2)

arable_land_df,arable_land_df_trnsps = read_data('arable_land.xlsx')
gdp_df,gdp_df_trnsps = read_data('GDP_filtered.xlsx')
population_df,population_df_trnsps = read_data('population.xlsx')

arable_land_year = filter_year_dataframe(arable_land_df)
gdp_df_year = filter_year_dataframe(gdp_df)
population_df_year = filter_year_dataframe(population_df)

gdp_country_df = filter_dataframes(gdp_df_year)
population_country_df = filter_dataframes(population_df_year)

df_fit_trial = pd.DataFrame()
# df_fit = adult_literacy_year[2000].copy()
# df_fit = adult_literacy_year[2014].copy()
df_fit_trial['2000'] = arable_land_year[2000]
df_fit_trial['2014'] = arable_land_year[2014]
# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_fit to affect df_fish. This make the plots with the
# original measurements
df_fit_trial = norm_df(df_fit_trial)
print(df_fit_trial.describe())
print()

for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit_trial)

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(df_fit_trial, labels))

# Plot for four clusters
kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(df_fit_trial)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))

# colour map Accent selected to increase contrast between colours
plt.scatter(df_fit_trial['2000'], df_fit_trial['2014'],
            c=labels, cmap="Accent")

# show cluster centres
for ic in range(2):
    xc, yc = cen[ic, :]
    plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel("2000", size=12)
plt.ylabel("2014", size=12)
plt.title("Adult Literacy year(2000 vs 2014)", size=16)
plt.show()

cluster_df = arable_land_year
cluster_df['Cluster'] = labels

cluster_country_df = filter_dataframes(cluster_df)

# Create a bar graph

gdp_country_df.columns.astype(int)
df = pd.DataFrame({'2000': gdp_country_df[2000.0],
                   '2002': gdp_country_df[2002.0],
                   '2004': gdp_country_df[2004.0],
                   '2006': gdp_country_df[2006.0],
                   '2008': gdp_country_df[2008.0],
                   '2010': gdp_country_df[2010.0],
                   '2012': gdp_country_df[2012.0],
                   '2014': gdp_country_df[2014.0]}, index=gdp_country_df.index)
df.plot.bar()
plt.xlabel('Countries', size=12)
plt.ylabel('GDP', size=12)
plt.title('GDP', size=16)

# Show the graph
plt.show()

gdp_df_trnsps['years']=gdp_df_trnsps.index.values
yearss = gdp_df_trnsps['years']
gdp_per_capita = gdp_df_trnsps['China']


# import data

# define fitting function
def gdp_fit(x, a, b, c):
    return a*x**2 + b*x + c

# perform curve fit
params, cov = curve_fit(gdp_fit, yearss, gdp_per_capita)

# plot data and fitted curve
plt.scatter(yearss, gdp_per_capita, label="data")
plt.plot(yearss, gdp_fit(yearss, *params), label="fit", color='r')
plt.xlabel("Year")
plt.ylabel("GDP per capita")
plt.legend()
plt.show()
