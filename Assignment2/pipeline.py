# Machine Learning for Public Policy
# Assignment 2: Building an initial pipeline
# Name: Vi Nguyen

import matplotlib.pylab as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Key constant to limit the # of x-values labeled in histogram charting
MAX_X_UNIQUE_VALUES = 100

def read_explore_data(filename, filetype, output_filename, test_train_ind):
    '''
    Takes in a filename, and its respective filetype to convert to a pandas 
    dataframe. Currently takes in the following filetype: csv

    Returns 
    -the dataframe of the data, 
    -a dataframe of summary statistics per variable including mean, median, 
     mode, standard deviation, and the number of missing values for each 
     field

     Creates 
     -histograms of all variables to study distributions
    '''
    if filetype == 'csv':
        df = pd.read_csv(filename)
    else:
        print('only csv filetype is currently supported')

    # Key lists and dictionaries for use when calculating the dataframe of 
    # summary statistics
    desc_list = ['mean', 'median', 'mode', 'std dev', 'no. missing vals']
    stats_dict = {'mean': df.mean(), 'median': df.median(), 'mode':
    df.mode(), 'std dev': df.std(), 'no. missing vals': df.isnull().sum()}
    # Creates blank dataframe filled with NaN to store the summary statistics
    # that we'll calculate
    desc_df = pd.DataFrame(np.nan, index = desc_list, columns = list(df.columns))

    for col in df.columns:

        ## Plots the histograms
        plt.clf()
        df_hist = df[col].value_counts()
        x_vals = len(df_hist)
        if x_vals > MAX_X_UNIQUE_VALUES:
            print('{} contains {} values: too many to chart'.format(col, x_vals))
        elif df.groupby(col).size().empty == True:
            print('{} is empty'.format(col))
        else:
            title = 'Histogram: {} ({})'.format(col, test_train_ind)
            ax = df.groupby(col).size().plot.bar()
            ax.set_ylabel('Count of Occurances')
            ax.set_title(title)
            plt.setp(ax.get_xticklabels(), rotation = 'vertical', fontsize = 8)
            fig = ax.get_figure()
            png_name = 'hist_' + col + '_' + test_train_ind + '.png'
            fig.savefig(png_name)
            print('{} created'.format(png_name))

         ## Calculates the summary statistics
        for keyword, val_series in stats_dict.items():
            if col in val_series:
                if keyword == 'mode':
                    mode_str = ''
                    for val in val_series[col].values:
                        val_str = str(val)
                        if val_str not in mode_str:
                            if mode_str != '':
                                mode_str = mode_str + ', ' + val_str
                            else:
                                mode_str = val_str
                    desc_df.loc[keyword, col] = mode_str
                else: 
                    desc_df.loc[keyword, col] = val_series[col] 
    desc_df.to_csv(output_filename)
    
    print('{} created'.format(output_filename))

    return desc_df, df

# for when histograms are not useful
def boxplot(df, col, test_train_ind):
    plt.clf()
    df.boxplot(column = col)
    plt.title('Boxplot: {} ({})'.format(col, test_train_ind))
    png_name = 'boxplot_' + col + '_' + test_train_ind + '.png'
    rcParams.update({'figure.autolayout': True})
    plt.savefig(png_name)
    print('{} created'.format(png_name))


## Fill in missing data
def fillna_mean(dataframe, output_filename):
    ''' 
    Takes in a dataframe and a string for the output filename and creates 
    a csv of the updated dataset with the missing values filled in with the
    column's mean
    '''
    dataframe = dataframe.fillna(dataframe.mean())
    dataframe.to_csv(output_filename)
    print('{} created'.format(output_filename))

    return dataframe


def fillna_cond_mean(dataframe, list_of_attributes, cond_attribute, output_filename):
    ''' 
    Takes in a dataframe, a list of columns for which we we will fill in the 
    missing values for, a string or list of strings of the conditional 
    attributes that we want to use, and the output filename.
    Returns a csv of the updated dataset with the missing values filled in with the
    column's mean
    '''
    for attr in list_of_attributes:
        dataframe[attr] = dataframe.groupby(cond_attribute).\
        transform(lambda x: x.fillna(x.mean()))
    dataframe.to_csv(output_filename)
    print('{} created'.format(output_filename))

    return dataframe


##Processing category (to binary) and continuous (to discrete) data
def cont_to_discrete(dataframe, variable_name, no_bins, labels = None):
    '''
    Takes in a dataframe, and the name of a categorical variable that we need 
    to turn into discrete variables, the number of categories we want, and 
    an optional input of labels for cases where we want to label the categories 
    ourselves.
    Returns a dataframe with the column with continous variable replaced with
    categories
    '''
    values_to_turn_discrete = list(dataframe[variable_name])
    discrete_values = pd.cut(values_to_turn_discrete, no_bins, labels)
    print(len(discrete_values))
    dataframe[variable_name] = discrete_values

    return dataframe


def cat_to_binary(dataframe, variable_name = None):
    '''
    Takes in a dataframe, the name of a categorical variable that we need to 
    turn to a binary, and creates dummy variables for each value of the variable 
    Returns a dataframe with binary/dummy variables added.
    '''
    df = pd.get_dummies(dataframe, columns = variable_name)
    
    return df


def logreg(dataframe_train, dataframe_test, y_var):
    model = LogisticRegression()
    y_train = np.ravel(dataframe_train[y_var])
    X_train = dataframe_train.drop(y_var, axis = 1)
    model = model.fit(X_train, y_train)
    X_test = dataframe_test.drop(y_var, axis = 1)
    y_test = model.predict(X_test)
    print('{} accuracy score'.format(model.score(X_test, y_test)))
    print('{} positives'.format(y_train.mean()))
    # looking at coefficients
    coef = pd.DataFrame(list(zip(X_train.columns, np.transpose(model.coef_))))
    coef.columns = ['Features', 'Logistic Regression Coefficients']
    output_filename = 'coefficients.csv'
    coef.to_csv(output_filename)
    print('{} created'.format(output_filename))
    print(coef)
