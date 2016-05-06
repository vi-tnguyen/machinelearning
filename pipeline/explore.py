from readto_pd_df import read
import matplotlib.pylab as plt
from matplotlib import rcParams

def explore_stats(df):
	'''
	inputs: dataframe
	prints key statistics and graphs
	'''
	print(df.describe())

	# Key lists and dictionaries for use when calculating the dataframe of 
    # additional summary statistics
    desc_list = ['median', 'mode', 'no. missing vals']
    stats_dict = {'median': df.median(), 'mode': df.mode(), 'no. missing vals': df.isnull().sum()}
    # Creates blank dataframe filled with NaN to store the summary statistics
    # that we'll calculate
    desc_df = pd.DataFrame(np.nan, index = desc_list, columns = list(df.columns))
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
    print(desc_df)


def explore_hist(df):
	'''
	inputs: dataframe
	prints graphs (best used in jupyter notebook for inline display of charts)
	'''
	for col in df.columns:
		plt.clf()
		plt.figure()
		df[col].hist()

def explore_scatter(df, list_of_tuples):
	'''
	inputs: dataframe; list of tuples for x and y values of scatter plot
	prints graphs (best used in jupyter notebook for inline display of charts)
	'''
	for val in list_of_tuples:
		plt.clf()
		plt.figure()
		x_vals = val[0]
		y_vals = val[1]
		df.plot(kind = 'scatter', x = x_vals, y = y_vals)