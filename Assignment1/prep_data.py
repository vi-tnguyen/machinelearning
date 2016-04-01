# Machine Learning for Public Policy
# Assignment 1: Prepping Student Data
# Name: Vi Nguyen


import pandas as pd
import numpy as np


def descriptives(filename):
    '''
    Takes in a dataframe and returns a dataframe of summary statistics per 
    variable including mean, median, mode, standard deviation, and the number 
    of missing values for each field.
    '''
    df = pd.read_csv(filename)
    desc_list = ['mean', 'median', 'mode', 'std dev', 'no. missing vals']
    stats_dict = {'mean': df.mean(), 'median': df.median(), 'mode':
    df.mode(), 'std dev': df.std(), 'no. missing vals': df.isnull().sum()}

    #print(stats_dict)

    desc_df = pd.DataFrame(np.nan, index = desc_list, columns = list(df.columns))

    for col in list(df.columns):
        for keyword, val_series in stats_dict.items():
            print('column', col)
            print('value', desc_df.loc[keyword, col])
            if col in val_series:
                if keyword == 'mode':
                    mode_str = ''
                    for val in val_series[col].values:
                        val_str = str(val)
                        if (val_str not in mode_str) and (val_str != 'nan'):
                            if mode_str != '':
                                mode_str = mode_str + ', ' + val_str
                            else:
                                mode_str = val_str
                    desc_df.loc[keyword, col] = mode_str
                else: 
                    desc_df.loc[keyword, col] = val_series[col] 
    print(desc_df)
    return desc_df


#def infer_gender(dataframe):
    '''
    Takes in a dataframe with the data where the gender is missing and infers
    the gender of the student based on the student's name using the 
    genderize API from www.genderize.io.
    '''

#def cond_mean(dataframe, attribute):

#def regression(dataframe, attribute):

#def infer_quant_var(dataframe, attribute, function):
    '''
    Takes in a dataframe and an attribute (column) in the dataframe to fill
    in the missing values for that column using the function (the mean, 
    conditional mean, mode, etc. for the column.
    Returns a new dataframe with the missing values filled in.
    '''

