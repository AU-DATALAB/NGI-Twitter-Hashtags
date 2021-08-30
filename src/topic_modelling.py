'''
This script takes a list of twitter datasets and performs topic modelling on their content. It returns most probable keywords for each topic, as well as word clouds for each topic.
'''


# Path and system operations
import sys,os
import preprocessor as p
import pandas as pd
import pickle

# Adding utils
sys.path.append(os.path.join(".."))
from utils import tweet_lda_utils

from gensim.test.utils import datapath
import pickle 
from wordcloud import WordCloud

import spacy
import en_core_web_sm
nlp = en_core_web_sm.load(disable=["ner"])

import random
from collections import Counter

# Regex tools
import re
import string

#import nltk
#import ssl

#try:
#    _create_unverified_https_context = ssl._create_unverified_context
#except AttributeError:
#    pass
#else:
#    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download('stopwords')


# LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

from gensim.test.utils import datapath
from gensim.utils import simple_preprocess


# Visualization 
# import pyLDAvis.gensim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse 

# Setting path
path=os.path.join("..","..","data-twitter","combined_datasets")

# for ds in os.listdir(path):
for ds in os.listdir(path):
    if ds.endswith(".csv"):
        basename = os.path.splitext(os.path.basename(ds))[0]
        print(f"Currently processing dataset {basename}")
        tmp_path = os.path.join("..","..","data-twitter","combined_datasets", ds)
        data = pd.read_csv(tmp_path)
        
        # Extract hashtag
        hashtag = re.search('ANJA_B_(.*)_combined.csv', ds).group(1)
        
        # Creating folder
        # Creating new directories cointaining weights
        if not os.path.exists("topic_weights"):
            os.makedirs("topic_weights")
        if not os.path.exists(os.path.join("topic_weights", f"{hashtag}")):
            os.makedirs(os.path.join("topic_weights", f"{hashtag}"))    
        
        # Filter english tweets 
        english_tweets = data[data["lang"]=="en"]
        # Filter text
        text = english_tweets[["text"]]
 
        # Remove NAs and duplicates
        text = text.dropna()
        text.drop_duplicates(subset=None, keep="first", inplace=True)
        
        # Cleaning text using preprocessor tweets
        # URL's
        # Mentions
        # Hashtags
        # Emojis
        # Smileys
        # Spefic words etc..
        p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.MENTION, p.OPT.NUMBER)
        
        cleaned = [p.clean(row) for row in text['text']]
        cleaned_stopwords = [tweet_lda_utils.remove_things(row) for row in cleaned]
        
        # Make a corpus
        text_processed, id2word, corpus = tweet_lda_utils.make_corpus(cleaned_stopwords, nlp)
        
        # Creating evaluating 
        info, optimal_n = tweet_lda_utils.compute_metrics("coherence", 
                                                      basename,
                                                      id2word, 
                                                      corpus, 
                                                      text_processed, 
                                                      limit=50, 
                                                      start=10, 
                                                      step=10)
        
        print("[INFO] Performing model with optimal topic number...")
        lda_model = gensim.models.LdaMulticore(corpus=corpus,    # vectorized corpus - list of lists of tuples
                           id2word=id2word,  # gensim dict - mapping word to IDS
                           num_topics=optimal_n,    # number of topics
                           random_state=100, # set for reproducibility
                           eta = "auto",
                           alpha = "asymmetric",
                           chunksize=1000,     # batch data for efficiency
                           passes=1,        # number of full passes over data (similar to epochs in nn)
                           iterations=25,   # number of times going over single document (rather than corpus)
                           per_word_topics=True, #define word distributions
                           minimum_probability=0.0) #minimum value (so it also returns those that do not appear)
                            
        
                    
        
        print("[INFO] Extracting topic words...")
        df_topic_keywords = tweet_lda_utils.format_topics_sentences(ldamodel=lda_model, 
                                                      corpus=corpus, 
                                                      texts=text_processed)

        # Format
        df_dominant_topic = df_topic_keywords.reset_index()
        df_dominant_topic.columns = ['Chunk_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
        sent_topics_sorteddf = pd.DataFrame()

        # Group by dominant topic so that these are in order
        sent_topics_outdf_grpd = df_topic_keywords.groupby('Dominant_Topic')

        # For chunk number and rest of the columns in the dataframe grouped by dominant topic
        for i, grp in sent_topics_outdf_grpd:
            # Concatenate the chunks by dominant topic along the percentages contribution column (in ascending form).
            sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, 
                                              grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], # Choosing the one with highest percentage contribution
                                              axis=0)

        # Reset Index    
        sent_topics_sorteddf.reset_index(drop=True, inplace=True)

        # Format - instead of having the 
        sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

        sent_topics_sorteddf.to_csv(os.path.join("dominant_topics", f"{basename}_dominant_topics.csv"))
        

        
        print("[INFO] Extracting word weights and making wordclouds")
        # Wordclouds
        for t in range(lda_model.num_topics):
            # Saves in the folder with the specific hashtag 
            open_file = open(os.path.join("topic_weights",
                                          f"{hashtag}", 
                                          f"weights_{hashtag}_{t}.pkl"), "wb")
            # Using pickle
            pickle.dump(dict(lda_model.show_topic(t, 100)), open_file)
            open_file.close()

            # Plotting wordcloud and saving it
            plt.figure()
            plt.figure(figsize=(10, 8))

            plt.imshow(WordCloud(background_color="white", 
                                 colormap = "tab20", 
                                 min_font_size = 6).fit_words(dict(lda_model.show_topic(t, 100))))
            plt.axis("off")
            plt.title("Topic " + str(int(t)) + " in #" + hashtag)

            # Saves in the folder with the specific hashtag 
            plt.savefig(os.path.join("topic_weights", 
                                     f"{hashtag}", 
                                     f"wordcloud_{hashtag}_{t}.png"))
            plt.show()

                    
                    