import umap
import hdbscan
import pandas  as pd

def generate_embeddings(embeddings):

    n_neighbors = [2, 5, 10, 20, 50, 100, 200]

    min_dist = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]

    full_umap = {}

    param_df = pd.DataFrame(columns=["n_neighbors", "min_dist"])


    #iterating through n_neighbors and min_dist
    for nn in n_neighbors:

        for md in min_dist:

            umap_embeddings = (umap.UMAP(n_neighbors = nn,
                                 min_dist = md, random_state = 42, n_components = 96).fit_transform(embeddings))

            full_umap["n_neighbors = {nn}, min_dist = {md}.obj".format(nn = nn, md =md)] = umap_embeddings
            param_df = param_df.append({"n_neighbors": nn, "min_dist": md}, ignore_index = True)


    return full_umap, param_df

def hdbscan_optim(full_umap, param_df, samp_size):
    #hdbscan params
    min_cluster_size =[int((samp_size/25)),int((samp_size/50)),int((samp_size/100)),int((samp_size/150)),int((samp_size/200)),int((samp_size/250))]

    min_samples = [int((samp_size/10)),int((samp_size/20)),int((samp_size/40)),int((samp_size/80)),int((samp_size/160)),int((samp_size/320)),int((samp_size/400)),int((samp_size/800))]

    results_df = pd.DataFrame(columns=["min_cluster_size", "min_samples", "DBCV"])

    def generate_clusters(embeddings, mcs, ms):        
        clusters = hdbscan.HDBSCAN(min_cluster_size = mcs,
                                   min_samples = ms,
                                   metric='euclidean', 
                                   cluster_selection_method='eom', gen_min_span_tree = True).fit(embeddings)

        return clusters


    for umap_embeddings_dict in full_umap.values():
        
        model_performance = pd.DataFrame(columns=["min_cluster_size", "min_samples", "DBCV"])
        
    #iterating through n_neighbors and min_dist
        for mcs in min_cluster_size:

            for ms in min_samples:

                clusters = generate_clusters(umap_embeddings_dict,
                  mcs,
                  ms)

                dbcv = clusters.relative_validity_
                
                model_performance = model_performance.append({"min_cluster_size": mcs, "min_samples": ms, "DBCV": dbcv}, ignore_index = True)
                
                
                
        max_dbcv_values = model_performance.loc[model_performance["DBCV"].idxmax()]

        dbcv_score = max_dbcv_values['DBCV']
        ms_best = max_dbcv_values['min_samples']
        mcs_best = max_dbcv_values['min_cluster_size']

        results_df = results_df.append({"min_cluster_size": mcs_best, "min_samples": ms_best, "DBCV": dbcv_score}, ignore_index = True) 
        
        del model_performance 

    full_results = pd.concat([param_df, results_df], axis =1, ignore_index = False)

    return full_results 