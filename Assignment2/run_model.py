import pipeline as pl
import matplotlib.pylab as plt
from matplotlib import rcParams

desc_df, df = pl.read_explore_data('cs-training.csv', 'csv', 'summary_stats_test.csv', 'train')
desc_df_test, df_test = pl.read_explore_data('cs-test.csv', 'csv', 'summary_stats_test.csv', 'test')
df = pl.fillna_mean(df, 'fillna_mean_train.csv')
df_test = pl.fillna_mean(df_test, 'fillna_mean_test.csv')


list_convert_to_discrete = ['RevolvingUtilizationOfUnsecuredLines', 
'DebtRatio', 'MonthlyIncome']

for col in list_convert_to_discrete:
    pl.boxplot(df, col, 'train')
    pl.boxplot(df_test, col, 'test')

pl.logreg(df, df_test, 'SeriousDlqin2yrs')