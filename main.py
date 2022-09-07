from utils import *
from data_preprocess import preprocess
from utils import get_topic_stats 
from utils import visualize_barchart_custom
from utils import topic_merging
from utils import set_filepath
from data_collating import collate_data
from hyperparameter_optimisation import generate_embeddings
from hyperparameter_optimisation import hdbscan_optim
import bertopic

MODEL_PATH = "/home/jovyan/TX_105_DynamicTopicModelling/sentence-transformers-all-mpnet-base-v2"

files = set_filepath(sys.argv[1])

#collating data from all files in the path
print("Collating data files... ")
feedback_data = collate_data(files)

docs = (feedback_data['all_comments'])
comments = docs.copy()

#preprocessing data
print("Preprocessing the data... ")
cleanlist = preprocess(comments)

#fitting sentence transformer  
model = SentenceTransformer(MODEL_PATH)

embedding_transformer = transformer_embedding(cleanlist)

# # #hyper parameter optim
print("Getting optimal hyperparameters... ")
full_umap, param_df = generate_embeddings(embedding_transformer)

samp_size = len(cleanlist)
parameter_values = hdbscan_optim(full_umap, param_df, samp_size)
print(parameter_values)
max_dbcv_values = parameter_values.loc[parameter_values["DBCV"].idxmax()]

best_dbcv = parameter_values["DBCV"].max()
print(f"Best DBCV score is, {best_dbcv}")

# # #Setting parameter values based on search results 
nn = int(max_dbcv_values['n_neighbors'])
md = int(max_dbcv_values['min_dist'])
ms = int(max_dbcv_values['min_samples'])
mcs = int(max_dbcv_values['min_cluster_size'])

print(f"nn: {nn}, type@ {type(nn)}\nmd: {md}, type@ {type(md)}\nms: {ms}, type@ {type(ms)}\nmcs: {mcs}, type@ {type(mcs)}")

print("Fitting the BERTopic model... ")
umap_model = umap.UMAP(n_neighbors = nn, min_dist = md, random_state = 42, n_components = 96)
umap_embeddings = umap_model.fit_transform(embedding_transformer)
hdbscan_model = hdbscan.HDBSCAN(min_samples = ms, min_cluster_size = mcs)
# umap_model = umap.UMAP()
# hdbscan_model = hdbscan.HDBSCAN(min_samples = 15, min_cluster_size = 10)
topic_model = BERTopic(umap_model = umap_model, hdbscan_model = hdbscan_model, nr_topics="auto", embedding_model=model, verbose = True)
topics, probabilities = topic_model.fit_transform(cleanlist)

#merging similar topics 
topics_to_merge = topic_merging(topic_model)

#if there are no topics to merge, ignore
if not topics_to_merge:
    pass
else:
    topic_model.merge_topics(cleanlist, topics, topics_to_merge)
    topics= topic_model._map_predictions(topic_model.hdbscan_model.labels_)


print("Producing Visualisations and summary statistics... ")

if os.path.exists("outputs") == True:
    pass
else:
    os.mkdir("outputs")


#topic statistics 
topic_info = topic_model.get_topic_info()
array = topic_model.hdbscan_model.labels_
topic_stats = get_topic_stats(topic_info, array, feedback_data)
topic_stats_fig = go.Figure(data=[go.Table(
    header=dict(values=list(topic_stats.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[topic_stats.AvgEase, topic_stats.AvgCharLength, topic_stats.Topic],
               fill_color='lavender',
               align='left'))
],
                           layout=go.Layout(
        title=go.layout.Title(text="Topic Statistics")))
topic_stats_fig.write_html("outputs/topic_stats.html")    

print(topic_info)

#getting topic info
topic_info = topic_model.get_topic_info()
topic_info_fig = go.Figure(data=[go.Table(
    header=dict(values=list(topic_info.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[topic_info.Topic, topic_info.Count, topic_info.Name],
               fill_color='lavender',
               align='left'))
],
                          layout=go.Layout(
        title=go.layout.Title(text="Topic Info")))
topic_info_fig.write_html("outputs/topic_info.html")

#topics over time 
timestamps = feedback_data['Date']
topics_over_time = topic_model.topics_over_time(cleanlist, topics, timestamps, nr_bins=20)
topics_over_time_vis = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
topics_over_time_vis.write_html("outputs/topics_over_time.html")

#top words per topic
easescores = topic_stats['AvgEase']
topic_barchart = visualize_barchart_custom(topic_model, easescores, top_n_topics=50)
topic_barchart.write_html("outputs/topic_barchart.html")

#topic hierarchy
hierarchical_topics = topic_model.hierarchical_topics(docs, topics)
hierarchy = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
hierarchy.write_html("outputs/topic_hierarchy.html")
print("Done! ")

#documents per topic
topic_docs = {topic: [] for topic in set(topics)}
for topic, doc in zip(topics, cleanlist):
    topic_docs[topic].append(doc)
    

if os.path.exists("topic_docs") == True:
    pass
else:
    os.mkdir("topic_docs")

if topic_info["Topic"].iloc[0] == -1:

    for i in range(-1, len(topic_info)-1):

        filename = "topic_docs/topic_docs%s.txt" % i

        with open(filename, 'w+') as f:
            for lines in topic_docs[i]:
                f.write("%s\n" % lines)

else:
    for i in range(len(topic_info)):

        filename = "topic_docs/topic_docs%s.txt" % i

        with open(filename, 'w+') as f:
            for lines in topic_docs[i]:
                f.write("%s\n" % lines)

#dbci score 

dbi = dbi_score(umap_embeddings, array)
print(f"DBI score for clustersd: ",dbi)