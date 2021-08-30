'''
This script loads a pertained emotion classifier and classifies the content of twitter datasets. It saves the final distribution of each dataset.
'''

# Data
import pandas as pd
import numpy as np
import re
import os,sys

# Bert
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model='bhadresh-savani/distilbert-base-uncased-emotion')

# Count dictionaries
from collections import Counter

# Status bar
from tqdm import tqdm

# Plot 
import matplotlib.pyplot as plt

# Adding utils
sys.path.append(os.path.join(".."))
from utils import tweet_lda_utils
import argparse 

def prep_emotion(ds):
    print("[INFO] Loading data...")
    # Cleaning the test dataframe
    # Filter english tweets 
    ds_en = ds[ds["lang"]=="en"]
    # Filter text
    text = ds_en[["text"]]

    # Remove NAs and duplicates
    text = text.dropna()
    text.drop_duplicates(subset=None, keep="first", inplace=True)
    
    # p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.MENTION, p.OPT.HASHTAG)

    # cleaned = [p.clean(row) for row in text['text']]
    # cleaned_stopwords = [tweet_lda_utils.remove_things(row) for row in cleaned]
    
    return text


def extract_emotion(text, name, args):
    print("[INFO] extracting emotions from tweets...")
    labels = []
    scores = []
    for row in tqdm(text['text']):
        prediction = classifier(row)
        labels.append(prediction[0]["label"])
        scores.append(prediction[0]["score"])

    text["bert_label"] = labels
    text["bert_scores"] = scores
    
    # Create dictionary
    dict_emotion = Counter(text["bert_label"])
    series_emotion = pd.Series(dict_emotion, name = "n_emotion")
    series_emotion.index.name = 'emotion_type'
    series_emotion = series_emotion.reset_index()
    series_emotion['hashtag'] = name
    
    # Save
    series_emotion.to_csv(os.path.join("stats", f"emotions_{name}.csv"))
    return text

def plot_emotion_pie(text, args, hashtag):
    print("[INFO] plotting emotions as piechart...")
    emotion_values = np.array(text.groupby("bert_label").agg("count")['text'])
    labels = list(set(text['bert_label']))
    colors = ['#66b3ff','#ff9999','#99ff99','#ffcc99', 'purple', 'green']

    # Circle plot

    fig1, ax1 = plt.subplots()
    ax1.pie(emotion_values, colors = colors, labels = labels, autopct='%1.1f%%', startangle=60)

    # Draw circle
    centre_circle = plt.Circle((0,0),0.60,fc='white')
    fig = plt.gcf()

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    plt.title(f"Distribution of emotions in #{hashtag}")
    
    plt.savefig(os.path.join("figs",f"emotion_pie_{hashtag}.png"), dpi=300, bbox_inches="tight")
    
    print(f'[INFO] network visualization saved in figs as emotion_pie_{hashtag}.png')
    
    plt.show()
    
    
def process(args):
    path = args['path']
    
    for ds in os.listdir(path):
    
        if ds.endswith(".csv"):
            name = re.search('ANJA_B_(.*)_combined.csv', ds)
            hashtag = name.group(1)

            print(f"Currently processing dataset {hashtag}")

            # Temporary path
            tmp_path = os.path.join(path, ds)

            hashtag_ds = pd.read_csv(tmp_path)

            text = prep_emotion(hashtag_ds)

            text = extract_emotion(text, hashtag, args)

            plot_emotion_pie(text, args, hashtag)

    
def main(): 
    # Add description
    ap = argparse.ArgumentParser(description = "[INFO] creating network pipeline") # Defining an argument parse

    ap.add_argument("-p","--path", 
                    required=False, # As I have provided a default name it is not required
                    type = str, # Str type
                    default = os.path.join("..", "..", "data-twitter", "combined_datasets"), # Setting default to the name of my own edgelist
                    help = "str of filename location")

    
    args = vars(ap.parse_args())
    
    process(args)
        
if __name__=="__main__":
    main()


