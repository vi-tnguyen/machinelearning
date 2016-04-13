import pipeline as pl

desc_df, df = pl.read_explore_data('cs-training.csv', 'csv', 'summary_stats.csv')
df = pl.fillna_mean(df, 'cs-training_fillna_mean.csv')
df = pl.cat_to_binary(df)


