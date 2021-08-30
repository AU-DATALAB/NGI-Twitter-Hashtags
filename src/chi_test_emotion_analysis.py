'''
This script takes emotion distribution of various testsets and tests whether they are significantly different than a baseline condition. It further tests differences within the datasets using adjusted residuals.
'''

import pandas as pd
import os
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import norm

# Status bar
from tqdm import tqdm
# Plot 
import matplotlib.pyplot as plt
import random
import math

# Change scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Baseline
## Combining files to one and saving to baseline
dirs = os.listdir(os.path.join("sentiment_analysis", "stats"))
baseline = pd.DataFrame()

for file in dirs:
    hashtag = pd.read_csv(os.path.join("sentiment_analysis", "stats", f"{file}"), index_col=0)
    print(file)
    sum(hashtag.n_emotion)

    baseline=baseline.append(hashtag)


emotions_summarized = baseline.groupby("emotion_type").sum("n_emotion").reset_index()
emotions_summarized['percent'] = (emotions_summarized['n_emotion']/emotions_summarized['n_emotion'].sum()*100).round(decimals=2).astype(str)

# Chi square test 
pivot_df = baseline.pivot(index="emotion_type", columns="hashtag", values="n_emotion")
pivot_df['baseline'] = emotions_summarized['n_emotion'].values

ph_res_total = pd.DataFrame()

for col in pivot_df.columns[0:10]:
    print(col)
    # Make all rows in every column percentages
    # pivot_df = pivot_df.apply(lambda row : (row/row.sum()*100).round(decimals=2), axis=0)
    test_case = pivot_df[["baseline",col]]

    # V2
    cross_table = test_case[["baseline",col]][0:6]
    chi2, p, df, exp = chi2_contingency(cross_table, correction=False) # chi2: the test statistic, p:the p-value of the test, dof:degrees of freedom
    if p < 0.05/10:
        # Column sums
        col_totals = cross_table.sum()
        # Number of rows
        ncols = len(col_totals)

        # Row sumns
        row_totals = cross_table.sum(axis=1)
        # Number of rows
        nrows = len(row_totals)

        # Grand sum
        n = sum(row_totals)

        tmp_ph_res = pd.DataFrame(columns=['emotion_type', 'hashtag', 'adj_res'])
        # We are going over all the cells (all rows and columns)
        for row in range(nrows):
            for column in range(ncols):
                # Take the observed count
                # Subtract the expected count 
                # Divide by the expected count times 1 minus row total divided by the grand sum times 1 minus column total divided by the grand sum raised to the power of .5 
                adj_res = (cross_table.iloc[row,column] - exp[row,column])/(exp[row,column]*(1-row_totals[row]/n)*(1-col_totals[column]/n))**0.5 # **0.5 is raising to the power of .5 is the same as the square root
                # Create a dataframe with the categories, the adjusted residuals relate to
                tmp_ph_res = tmp_ph_res.append({'emotion_type':cross_table.index[row], 'hashtag':cross_table.columns[column], 'adj_res':adj_res}, ignore_index=True)

        # The adjusted residuals are actually z-values - z-values can be transformed into a normal distribution and therfore we can calculate probabilities --> p-values
        # Multiply by 2 to make it a 2-tailed t-test
        # These are corrected with bonferroni 
        # Significance test and adjusted residuals
        tmp_ph_res['sig'] = 2*(1-norm.cdf(abs(tmp_ph_res['adj_res']))) # Getting the probability
        
        # Adjust for multiple testing
        tmp_ph_res['adj_sig'] = tmp_ph_res.shape[0]*tmp_ph_res['sig']
        tmp_ph_res['condition'] = col
        ph_res_total = ph_res_total.append(tmp_ph_res)
    else:
        print(f"Chi test for {col} is not significant")

print(ph_res_total)

ph_res_total = ph_res_total[ph_res_total.hashtag!="baseline"]
ph_res_total['adj_res'] = round(ph_res_total['adj_res'], ndigits=2)
ph_res_total['adj_sig'] = round(ph_res_total['adj_sig'], ndigits=2)
ph_res_total.to_csv("emotions_hashtags_baseline.csv")

# %%
# Filtering significant emotions below the threshold:
significant_emotions = ph_res_total[ph_res_total.adj_sig<0.05]
significant_emotions = significant_emotions[significant_emotions.hashtag!="baseline"]

significant_emotions.to_csv("significant_emotions_hashtags_baseline.csv")
# Bonferroni
# Our first test at the level 0.05 is a 5% chance of a false positive; the test after that would be 10% chance of a false positive, and so forth. With each subsequent test, one would be increasing the error rate by 5%. 
