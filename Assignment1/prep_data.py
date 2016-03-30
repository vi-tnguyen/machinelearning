# Machine Learning for Public Policy
# Assignment 1: Prepping Student Data
# Name: Vi Nguyen


import pandas as pd

mock_stnt_data = pandas.read_csv('mock_student_data.csv')
mock_student_data.describe

def descriptives(dataframe):
    '''
    Takes in a dataframe and returns a list of tuple of summary statistics 
    including mean, median, mode, standard deviation, and the number of missing 
    values for each field.
    '''
    
    desc_dict = {}
    #desc = dataframe.describe().transpose()
    for col in df.columns
    df['mean'] = df.mean()




def infer_gender(dataframe):
    '''
    Takes in a dataframe with the data where the gender is missing and infers
    the gender of the student based on the student's name using the 
    genderize API from www.genderize.io.
    '''

def cond_mean(dataframe, attribute):

def mean(dataframe, attribute):

def mode(dataframe, attribute):

def regression(dataframe, attribute):

def infer_quant_var(dataframe, attribute, function):
    '''
    Takes in a dataframe and an attribute (column) in the dataframe to fill
    in the missing values for that column using the function (the mean, 
    conditional mean, mode, etc. for the column.
    Returns a new dataframe with the missing values filled in.
    '''

