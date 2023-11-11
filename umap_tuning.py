import optuna
import main as M
from src import config as C
from src import utils as U
from collections import defaultdict
from scipy import spatial
from scipy.stats import spearmanr
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_cos_dist(people, embeddings):
    """
    given the list of people and their description embeddings, return the cosine distance of each 
    person with all people in the list sorted by name of the person in alphabetical order.
    """
    top_matches = {}
    all_personal_pairs = defaultdict(list)
    for person in people:
        for person1 in people:
            all_personal_pairs[person].append([spatial.distance.cosine(embeddings[person1], embeddings[person]), person1])

    for person in people:
        top_matches[person] = sorted(all_personal_pairs[person], key=lambda x: x[1])
    return top_matches


def get_euclidean_dists(people, umapVec):
    """
    caculate the list of euclidean distances of each person with all people in the list based on their plot point on the umap vector
    """
    res = {}
    all_personal_pairs = defaultdict(list)
    for person in people:
        for person1 in people:
            all_personal_pairs[person].append([math.dist(umapVec[person1], umapVec[person]), person1])

    for person in people:
        res[person] = sorted(all_personal_pairs[person], key=lambda x: x[1])
    return res


def convert_to_df(data_dict):
    """
    convert the data dictionaries to dataframes for faster spearmanr calculation
    """
    df = pd.DataFrame(data=data_dict)
    df.index = df['Greg Kirczenow'].apply(lambda arr: arr[1])
    df.index.name = None
    for col in df.columns:
        df[col] = df[col].apply(lambda arr: arr[0])
    return df

### Optuna tuning
def objective(trial, mm, cos_dist_flatten, random_state):
    """
    objective function to be minimized.
    """
    #hyper params to be optimized
    n_neighbors = trial.suggest_int("n_neighbors", 2, 53)
    min_dist = trial.suggest_float("min_dist", 0, 0.99)
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine'])

    #calculate umap vector and the corresponding distances
    umapVec = mm.dimension_reduction(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
    euclid_distances = get_euclidean_dists(mm.data.keys(), umapVec)
    euclid_dist_matrix = convert_to_df(euclid_distances)
    euclid_dist_flatten = euclid_dist_matrix.values.flatten()
    corr = spearmanr(cos_dist_flatten, euclid_dist_flatten)
    corr_inv = 1-corr.statistic
    return corr_inv


def calc_correlation(mm, cos_dist_flatten, umapVec, title):
    euclid_distances = get_euclidean_dists(mm.data.keys(), umapVec)
    euclid_dist_matrix = convert_to_df(euclid_distances)
    # euclid_dist_matrix = euclid_dist_matrix.reindex(sorted(euclid_dist_matrix.columns), axis=1)
    euclid_dist_flatten = euclid_dist_matrix.values.flatten()
    corr = spearmanr(cos_dist_flatten, euclid_dist_flatten)
    plt.figure(figsize=(14, 12))
    plt.title(title + ' cosine distance and euclid distance Spearman correlation')
    a, b = np.polyfit(cos_dist_flatten, euclid_dist_flatten, 1)
    plt.scatter(cos_dist_flatten, euclid_dist_flatten)
    plt.plot(cos_dist_flatten, a*cos_dist_flatten+b, 'k')
    plt.legend(title=f'r={corr.statistic}, pvalue={corr.pvalue}', frameon=False)
    plt.xlabel('cosine distance')
    plt.ylabel('euclidean distance')
    plt.savefig(title+'_cosine_euclid_corr.png')
    return corr


def main():
    mm = M.MatchMaker(C.FILE_NAME, C.MINILM_L6_V2)
    mm.make_pipeline(preprocess=True, embed_sentence=True)
    umapVec = mm.dimension_reduction()
    mm.visualization('umap_untuned.png')

    #### pre-tuning
    top_matches = get_cos_dist(mm.data.keys(), mm.embeddings)
    cos_dist_matrix = convert_to_df(top_matches)
    # cos_dist_matrix = cos_dist_matrix.reindex(sorted(cos_dist_matrix.columns), axis=1) 
    cos_dist_flatten = cos_dist_matrix.values.flatten()
    pretuned_corr = calc_correlation(mm, cos_dist_flatten, umapVec, 'pretuned')
    print('pretuned correlation:', pretuned_corr)
    
    #### tuning using optuna
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, mm, cos_dist_flatten, C.RANDOM_STATE), n_trials=100)
    print(f"\nBest parameters: {study.best_params}.  Highest Spearmanr: {1 - study.best_value}")

    umap_tuned = mm.dimension_reduction(n_neighbors=study.best_params['n_neighbors'], min_dist=study.best_params['min_dist'], 
                                        metric=study.best_params['metric'], random_state=C.RANDOM_STATE)
    tuned_corr = calc_correlation(mm, cos_dist_flatten, umap_tuned, 'tuned')
    print(tuned_corr)
    mm.visualization('umap_tuned.png')


if __name__ == "__main__":
    main()


