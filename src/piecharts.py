'''
This script takes a directory containing distribution files with different distributions of emotions and transforms these into a visualisation of piecharts.
'''


import pandas as pd
import os
import numpy as np
from scipy.stats import chi2_contingency
# Status bar
from tqdm import tqdm
# Plot 
import matplotlib.pyplot as plt
import random
    

# Baseline
## Combining files to one and saving to baseline
dirs = os.listdir(os.path.join("sentiment_analysis", "stats"))
baseline = pd.DataFrame()

for file in dirs:
    hashtag = pd.read_csv(os.path.join("sentiment_analysis", "stats", f"{file}"), index_col=0)
    baseline=baseline.append(hashtag)

emotions_summarized = baseline.groupby("emotion_type").sum("n_emotion").reset_index()
emotions_summarized['percent'] = (emotions_summarized['n_emotion']/emotions_summarized['n_emotion'].sum()*100).round(decimals=2).astype(str)



# Further processing and piechart making of base line
# hashtag['labels'] = hashtag[['emotion_type', 'percent']].agg(': '.join, axis=1)
emotions_summarized['labels'] = emotions_summarized['emotion_type']+ ': ' + emotions_summarized['percent'].astype(str) + '%'

fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(aspect="equal"))

labels = list(emotions_summarized['labels'])
data = emotions_summarized['n_emotion']
colors = ['#66b3ff','#ff9999','#99ff99','#ffcc99', 'purple', 'green']

wedges, texts = ax.pie(data, startangle=-40, colors = colors, labeldistance=1.5,wedgeprops = { 'linewidth' : 1.5, 'edgecolor' : 'white' }) # ax.pie both returns a sequence of matplotlib.patches.Wedge instances and alist of the label matplotlib.text.Text instances

# bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
        # bbox=bbox_props, 
        zorder=0, va="center")
for i, p in enumerate(wedges): 
    ang = (p.theta2 - p.theta1)/2 + p.theta1 
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = f"angle,angleA=1,angleB={ang}"
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    
    ax.annotate(labels[i], xy=(x, y), xytext=(1.50*np.sign(x), 1.2*y),
            horizontalalignment=horizontalalignment, 
                **kw)
plt.tight_layout()
ax.set_title(f"Distribution of emotions in baseline")
plt.show()


## Running all plots within the interactive window
import pandas as pd
import os
import numpy as np

# Status bar
from tqdm import tqdm
# Plot 
import matplotlib.pyplot as plt
import re

dirs = os.listdir(os.path.join("sentiment_analysis", "stats"))

for file in dirs:
    name = re.search('emotions_(.*).csv', file)
    name = name.group(1)

    hashtag = pd.read_csv(os.path.join("sentiment_analysis", "stats", f"{file}"), index_col=0)
    
    hashtag.sort_values(by = 'emotion_type', axis=0, inplace= True)
    
    hashtag['percent'] = (hashtag['n_emotion']/hashtag['n_emotion'].sum()*100).round(decimals=2).astype(str)
   # hashtag['labels'] = hashtag[['emotion_type', 'percent']].agg(': '.join, axis=1)
    hashtag['labels'] = hashtag['emotion_type']+ ': ' + hashtag['percent'].astype(str) + '%'

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(aspect="equal"))
    
    labels = list(hashtag['labels'])
    data = hashtag['n_emotion']
    colors = ['#66b3ff','#ff9999','#99ff99','#ffcc99', 'purple', 'green']
   # colors = dict(zip(labels, colors[:len(labels)]))

    wedges, texts = ax.pie(data, startangle=-40, colors = colors, labeldistance=1.5,wedgeprops = { 'linewidth' : 1.5, 'edgecolor' : 'white' }) # ax.pie both returns a sequence of matplotlib.patches.Wedge instances and alist of the label matplotlib.text.Text instances

    # bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
            # bbox=bbox_props, 
            zorder=0, va="center")
    for i, p in enumerate(wedges): 
        ang = (p.theta2 - p.theta1)/2 + p.theta1 
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=1,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        
        ax.annotate(labels[i], xy=(x, y), xytext=(1.50*np.sign(x), 1.2*y),
                horizontalalignment=horizontalalignment, 
                    **kw)
    plt.tight_layout()
    ax.set_title(f"Distribution of emotions in #{name}")
    plt.show()

