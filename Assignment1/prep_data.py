# Machine Learning for Public Policy
# Assignment 1: Prepping Student Data
# Name: Vi Nguyen

import matplotlib.pylab as plt
import pandas as pd
import numpy as np
# Requests installed request library for API use, with command 
# 'sudo pip3 install requests'
import requests
# String used to capitalize gender received from API to align with style in 
# original csv
import string

# Key constant to limit the # of x-values shown in histogram charting
MAX_X_UNIQUE_VALUES = 100

# List of values for which we will infer missing values
list_of_attributes = ['Age', 'GPA', 'Days_missed']

def descr_hist(filename, output_filename):
    '''
    Takes in a dataframe and returns a dataframe of summary statistics per 
    variable including mean, median, mode, standard deviation, and the number 
    of missing values for each field.
    '''
    df = pd.read_csv(filename)

    # Key lists and dictionaries for use when calculating the dataframe of 
    # summary statistics
    desc_list = ['mean', 'median', 'mode', 'std dev', 'no. missing vals']
    stats_dict = {'mean': df.mean(), 'median': df.median(), 'mode':
    df.mode(), 'std dev': df.std(), 'no. missing vals': df.isnull().sum()}
    # Creates blank dataframe filled with NaN to store the summary statistics
    # that we'll calculate
    desc_df = pd.DataFrame(np.nan, index = desc_list, columns = list(df.columns))

    for col in list(df.columns):

        ## Plots the histograms
        plt.clf()
        df_hist = df[col].value_counts()
        title = 'Mock Student Data Histogram: ' + col
        ax = df_hist.plot(kind = 'bar', title = title)
        ax.set_ylabel('Count of Students')

        # Limits the number of x-values displayed in cases where the variable
        # has too many values (i.e. ID, Names)
        if len(df_hist) > MAX_X_UNIQUE_VALUES:
            plt.locator_params(nbins = MAX_X_UNIQUE_VALUES / 2, axis = 'x')
            new_title = title + '\n' + '(Only {} x-values are labeled)'.format(
                MAX_X_UNIQUE_VALUES // 2)
            ax.set_title(new_title)

        fig = ax.get_figure()
        png_name = 'hist_' + col + '.png'
        fig.savefig(png_name)
        print('{} created'.format(png_name))
        plt.show()

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


def infer_gender(dataframe, name_column, gender_column, output_filename):
    '''
    Takes in a dataframe with the data where the gender is missing, and the
    name column that we want to use to infer the gender of the student based 
    using the genderize API from www.genderize.io.
    '''
    MAX_NAMES_PER_QUERY = 10
    # Adds a country filter to localize the inference data set to just U.S names
    # and genders. Country code can be found here: 
    # https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
    COUNTRY = 'us'

    # Saves a series of rows in dataframe that had Gender as null; and retains
    # the index of the original dataframe for us to reference
    null_gender_srs = dataframe[dataframe[gender_column].isnull()]

    for i in null_gender_srs.index:
        indiv_name = df.loc[i, name_column]
        r = requests.get('https://api.genderize.io/?name={}&country_id={}'\
            .format(indiv_name, COUNTRY))
        dataframe.loc[i, gender_column] = string.capwords(r.json()['gender'])

    dataframe.to_csv(output_filename)
    print('{} created'.format(output_filename))
    print('Number of records with missing gender after inference: {}'.format(
        df[gender_column].isnull().sum()))

    return dataframe

def fillna_mean(dataframe, output_filename):
    dataframe = dataframe.fillna(dataframe.mean())
    dataframe.to_csv(output_filename)
    print('{} created'.format(output_filename))


def cond_mean(dataframe, list_of_attributes, cond_attribute, output_filename):
    for attr in list_of_attributes:
        dataframe[attr] = dataframe.groupby(cond_attribute).\
        transform(lambda x: x.fillna(x.mean()))
    dataframe.to_csv(output_filename)
    print('{} created'.format(output_filename))

#def regression(dataframe, attribute):

desc_df, df = descr_hist('mock_student_data.csv', 'summary_stats.csv')
df = infer_gender(df, 'First_name', 'Gender', 
    'mock_student_data_gender_inferred.csv')
fillna_mean(df, 'mock_student_data_gend_inf_fillna_mean.csv')
cond_mean(df, list_of_attributes, 'Graduated', 
    'mock_student_data_gend_inf_fillna_cond_mean.csv')
cond_mean(df, list_of_attributes, ['Graduated', 'Female'] 
    'mock_student_data_gend_inf_fillna_cond_mean2.csv')
