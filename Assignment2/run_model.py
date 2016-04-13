import pipeline as pl
import matplotlib.pylab as plt

desc_df, df = pl.read_explore_data('cs-training.csv', 'csv', 'summary_stats.csv')
df = pl.fillna_mean(df, 'cs-training_fillna_mean.csv')

list_convert_to_discrete = ['RevolvingUtilizationOfUnsecuredLines', 
'DebtRatio', 'MonthlyIncome']

for col in list_convert_to_discrete:
    df = pl.cont_to_discrete(df, col, 10)
    plt.clf()
    df_hist = df[col].value_counts()
    x_vals = len(df_hist)
    title = 'Histogram: ' + col
    ax = df_hist.plot(kind = 'bar', title = title)
    ax.set_ylabel('Count of Students')
    fig = ax.get_figure()
    png_name = 'hist_' + col + '.png'
    fig.savefig(png_name)
    print('{} created'.format(png_name))
    plt.show()





