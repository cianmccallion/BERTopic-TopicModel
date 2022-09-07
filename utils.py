import sys
import pandas as pd 
import glob 
import numpy as np
import re
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora 
from gensim.models import CoherenceModel
from gensim.models import ldaseqmodel
import artm
import spacy
import pickle
import pprint
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import umap
from sklearn.cluster import KMeans 
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import random
from word_forms.lemmatizer import lemmatize
import hdbscan
import umap
import itertools
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.metrics import davies_bouldin_score

data_path = r"./Data"
MODEL_PATH = "/home/jovyan/TX_105_DynamicTopicModelling/sentence-transformers-all-mpnet-base-v2"
model = SentenceTransformer(MODEL_PATH)

def set_filepath(sys_argv):
    if sys_argv == 'all':
        files = glob.glob(data_path + "/*.xlsx")
    elif sys_argv == 'sales':
        files = glob.glob(data_path + "/sales1-quote-journey-exit - Full extract.xlsx")
    elif sys_argv == 'mac-error':
        files = glob.glob(data_path + "/serv1-myaccount-error-pages - full extract.xlsx")
    elif sys_argv == 'mac-eof':
        files = glob.glob(data_path + "/serv4-mac-end-of-flow - full extract.xlsx")
    elif sys_argv == 'renewals':
        files = glob.glob(data_path + "/serv6-renewals-homepage - full extract.xlsx")
        
    return files 

def rename_ease(df, data_path):
    if data_path == ['./Data/sales1-quote-journey-exit - Full extract.xlsx', './Data/serv6-renewals-homepage - full extract.xlsx', './Data/serv1-myaccount-error-pages - full extract.xlsx', './Data/serv4-mac-end-of-flow - full extract.xlsx']:
        dict_ = {'MAC_End_EASE' : 'ease',
        'QJexit1_ease' : 'ease',
        'MyAccError1_Ease' : 'ease',
        'RN_HP_Ease' : 'ease'}
    elif data_path == ['./Data/sales1-quote-journey-exit - Full extract.xlsx']:
        dict_ = {'QJexit1_ease' : 'ease'}
    elif data_path == ['./Data/serv6-renewals-homepage - full extract.xlsx']:
        dict_ = {'RN_HP_Ease' : 'ease'}
    elif data_path == ['./Data/serv1-myaccount-error-pages - full extract.xlsx']:
        dict_ = {'MyAccError1_Ease' : 'ease'}
    elif data_path == ['./Data/serv4-mac-end-of-flow - full extract.xlsx']:
        dict_ = {'MAC_End_EASE' : 'ease'}
        
    df.rename(columns=dict_, inplace=True)
    
    return df
           
def join_columns(feedback_data, data_path):
    if data_path == ['./Data/sales1-quote-journey-exit - Full extract.xlsx', './Data/serv6-renewals-homepage - full extract.xlsx', './Data/serv1-myaccount-error-pages - full extract.xlsx', './Data/serv4-mac-end-of-flow - full extract.xlsx']:
        #list of all open text box columns to be concatenated 
        cols = ['QJexit1_exit_reason_other','QJexit1_Technical_issue_comment','QJexit1_Payment_method_comment','QJexit1_Site_usability_comment', 'QJexit1_More_info_comment', 'QJexit1_Other_comment',
        'MyAccError1_policytype_other', 'MyAccError1_motor_intent_other', 'MyAccError1_MAC_motor_other', 'MyAccError1_ATP_motor_other', 'MyAccError1_home_intent_other', 'MyAccError1_MAC_home_other',
        'MyAccError1_ATP_home_other', 'MyAccError1_comment_home', 'MyAccError1_comment_otherproduct',
        'RN_HP_Expectation', 'RN_HP_ConfusingText', 'RN_HP_LowEaseText', 
        'MAC_End_Easy_COM','MAC_End_Diff_COM'] 
    elif data_path == ['./Data/sales1-quote-journey-exit - Full extract.xlsx']:
        cols = ['QJexit1_exit_reason_other','QJexit1_Technical_issue_comment','QJexit1_Payment_method_comment','QJexit1_Site_usability_comment', 'QJexit1_More_info_comment', 'QJexit1_Other_comment']
    elif data_path == ['./Data/serv6-renewals-homepage - full extract.xlsx']:
        cols = ['RN_HP_Expectation', 'RN_HP_ConfusingText', 'RN_HP_LowEaseText']
    elif data_path == ['./Data/serv1-myaccount-error-pages - full extract.xlsx']:
        cols = ['MyAccError1_policytype_other', 'MyAccError1_motor_intent_other', 'MyAccError1_MAC_motor_other', 'MyAccError1_ATP_motor_other', 'MyAccError1_home_intent_other', 'MyAccError1_MAC_home_other',
        'MyAccError1_ATP_home_other', 'MyAccError1_comment_home', 'MyAccError1_comment_otherproduct']     
    elif data_path == ['./Data/serv4-mac-end-of-flow - full extract.xlsx']:
        cols = ['MAC_End_Easy_COM','MAC_End_Diff_COM']
    
    feedback_data['all_comments'] = feedback_data[cols].apply(lambda row: ' '.join(set(list(row.values.astype(str)))), axis = 1) # joining all open text boxes
    
    return feedback_data
    
#sentence transfromer 
def transformer_embedding(sentences):
    vec = np.array(model.encode(sentences, show_progress_bar = True))
    return vec


#Get topic summary statistics 
def get_topic_stats(topic_info, label_array, feedback_data):
    
    topics_mean_ease = []
    topics_mean_char_length = [] 
    #creating a list indices which relate the original documents to the topic 
    for i in range(len(topic_info)-1):

        indices = np.where(label_array == i)

        indices2 = np.asarray(indices)

        indices2 = indices2.tolist()

        flat_ls = []
        for j in indices2:
            for k in j:
                flat_ls.append(k)


        topics_docs = feedback_data.iloc[flat_ls]
        topics_mean_ease.append(topics_docs['ease'].mean())
        topics_mean_char_length.append(topics_docs['all_comments'].apply(len).mean())

        topic_stats = pd.DataFrame()
        topic_stats['AvgEase'] = topics_mean_ease
        topic_stats['AvgCharLength'] = topics_mean_char_length
        topic_stats['Topic'] = topic_stats.index
        topic_stats = topic_stats.round(2)

    return topic_stats


def visualize_barchart_custom(topic_model,
                       easescores,
                       topics: List[int] = None,
                       top_n_topics: int = 8,
                       n_words: int = 5,
                       width: int = 250,
                       height: int = 250) -> go.Figure:
    """ Visualize a barchart of selected topics
    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most frequent topics.
        n_words: Number of words to show in a topic
        width: The width of each figure.
        height: The height of each figure.
    Returns:
        fig: A plotly figure
    Usage:
    To visualize the barchart of selected topics
    simply run:
    ```python
    topic_model.visualize_barchart()
    ```
    Or if you want to save the resulting figure:
    ```python
    fig = topic_model.visualize_barchart()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/bar_chart.html"
    style="width:1100px; height: 660px; border: 0px;""></iframe>
    """
    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list()[0:6])

    # charcounts = topic_stats['AvgCharLength']
    
    
    # Initialize figure
    str1 = [f"Topic {topic}" for topic in topics]
    str2 = [f"Avg Ease Score: {ease}" for ease in easescores]
    #str3 = [f"Avg String Length: {length}" for length in charcounts]
    
    subplot_titles = list(map("<br>".join, zip (str1, str2,)))#str3)))
    
    columns = 4
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.6 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles)

    # Add barchart for each topic
    row = 1
    column = 1
    for topic in topics:
        words = [word + "  " for word, _ in topic_model.get_topic(topic)][:n_words][::-1]
        scores = [score for _, score in topic_model.get_topic(topic)][:n_words][::-1]

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': "<b>Topic Word Scores",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=16,
                color="Black")
        },
        width=width*4,
        height=(height*rows)+5 if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    fig.update_annotations(font_size=12)
    
    return fig


def topic_merging(topic_model):
    distance_matrix = cosine_similarity(np.array(topic_model.topic_embeddings)[1:, :])
    # labels = (topic_model.get_topic_info().sort_values("Topic", ascending=True).Name)[1:]

    similar = np.argwhere(distance_matrix > 0.85) #returns indices of all matrix values with a cosine similarity >0.85

    simlist = similar.tolist()

    def check(list):
        return list[0] == list[1] #returns true if both elements in list are equal (in this case the diagonals in the matrix)


    #creating list of indices where our min prob argument is reached ignoring the diagonals 
    simlist2 = []
    for i in simlist:
        output = check(i)
        if output == True: #ignores nested lists where both elements are equal i.e. the diagonal
            pass
        else:
            simlist2.append(i)

    for i in simlist2:
        i.sort()

        #removing duplicate indices i.e. ([24,32], [32,24])
    simset = set([tuple(l) for l in simlist2])
    topics_to_merge = [list(x) for x in simset]
    
    return topics_to_merge

def dbi_score(umap_embeddings, labels):
    dbi_score = davies_bouldin_score(umap_embeddings, labels)
    
    return dbi_score