import pandas as pd

## Fill in missing data
def fillna_mean(df):
    ''' 
    Takes in a dataframe and a string for the output filename and creates 
    a csv of the updated dataset with the missing values filled in with the
    column's mean
    '''
    for var in df.columns:
        print(var)
        if pd.isnull(df[var].mean()):
            df = df.drop(var, axis = 1)
            print('column dropped because it has no data: {}'.format(var))

    df = df.fillna(df.mean())

    return df


def fillna_cond_mean(df, list_of_attributes, cond_attribute, output_filename):
    ''' 
    Takes in a dataframe, a list of columns for which we will fill in the 
    missing values for, a string or list of strings of the conditional 
    attributes that we want to use, and the output filename.
    Returns a csv of the updated dataset with the missing values filled in with the
    column's mean
    '''
    for attr in list_of_attributes:
        df[attr] = df.groupby(cond_attribute).\
        transform(lambda x: x.fillna(x.mean()))
        print(df.groupby(cond_attribute)[attr].mean())

    return df


##Processing category (to binary) and continuous (to discrete) data
def cont_to_discrete(df, variable_name, no_bins, labels = None):
    '''
    Takes in a dataframe, and the name of a categorical variable that we need 
    to turn into discrete variables, the number of categories we want, and 
    an optional input of labels for cases where we want to label the categories 
    ourselves.
    Returns a dataframe with the column with continous variable replaced with
    categories
    '''
    values_to_turn_discrete = list(df[variable_name])
    discrete_values = pd.cut(values_to_turn_discrete, no_bins, labels)
    print(len(discrete_values))
    dataframe[variable_name] = discrete_values

    return df


def cat_to_binary(df, variable_name = None):
    '''
    Takes in a dataframe, the name of a categorical variable that we need to 
    turn to a binary, and creates dummy variables for each value of the variable 
    Returns a dataframe with binary/dummy variables added.
    '''
    df = pd.get_dummies(df, columns = variable_name)
    
    return df