import main as M
import matplotlib.pyplot as plt
from src import config as C
from src import utils as U
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

def num_rank_displaced(arr, displaceby, greater=True):
    """
    return number of rank placements being displaced if displacement is greater or less than displaceby parameter
    """
    if greater:
        count = [i for i in arr if abs(i) >= displaceby]
    else:
        count = [i for i in arr if abs(i) < displaceby]
    return len(count)

def print_top_5(mm: M.MatchMaker, top_5: list[list], target_person: str, model_name: str):
    print(f"\nTop five closest people to {target_person} by embeddings of {model_name}")
    print(f"\n{'Rank':<5} {'Name':<30} {'Cosine Similarity':<20} {'Description'}")
    for name, cosine_similarity, rank in top_5:
        print(f"{rank:<5} {name:<30} {cosine_similarity:<20.4f} {mm.data[name]}")

def compare_models(file_name: str, model_name_1: str, model_name_2: str, target_person: str, sub: int):
    """
    Compare two models from SentenceTransformer.
    """
    # file_name = C.FILE_NAME
    # model_name_1 = C.MINILM_L6_V2
    # model_name_2 = C.ALL_MPNET_BASE_V2
    # target_person = "Haodong Tao"
    # calculate embeddings for model 1
    mm = M.MatchMaker(file_name, model_name_1)
    mm.make_pipeline(preprocess=True, embed_sentence=True)
    top_matches_1 = U.similarity_ranking(mm.embeddings, target_person)

    # calculate embeddings for model 2
    mm.sentence_embedding(model_name_2)
    top_matches_2 = U.similarity_ranking(mm.embeddings, target_person)

    # x = [i for i in range(len(top_matches_1))]

    # calculate the spearmanr in cosine similarity and differences in ranking of everyone vs target person between the two models
    # cosine_diff = [top_matches_2[i][1] - top_matches_1[i][1] for i in range(len(top_matches_1))]
    model1_cossims = [top_matches_1[i][1] for i in range(len(top_matches_1))]
    model2_cossims = [top_matches_2[i][1] for i in range(len(top_matches_2))]
    model_corr = spearmanr(model1_cossims, model2_cossims)

    rank_diff = [top_matches_1[i][2] - top_matches_2[i][2] for i in range(len(top_matches_1))]
    num_displace_greater10 = num_rank_displaced(rank_diff, 10)
    num_displace_lesser5 = num_rank_displaced(rank_diff, 5, False)

    # print result
    print(f"{target_person}'s description: {mm.data[target_person]}")
    top_5 = sorted(top_matches_1, key=lambda x: x[2])[:5]
    print_top_5(mm, top_5, target_person, model_name_1)
    top_5 = sorted(top_matches_2, key=lambda x: x[2])[:5]
    print_top_5(mm, top_5, target_person, model_name_2)

    a, b = np.polyfit(model1_cossims, model2_cossims, 1)
    plt.subplot(2, 2, sub)
    plt.scatter(model1_cossims, model2_cossims)
    plt.plot(model1_cossims, a*np.array(model1_cossims)+b)
    plt.legend(title=f'r={model_corr.statistic}, pvalue={model_corr.pvalue}', frameon=False)
    plt.xlabel(model_name_1.split('/')[1])
    plt.ylabel(model_name_2.split('/')[1])
    # plt.savefig(model_name_1.split('/')[1]+'_vs_'+model_name_2.split('/')[1]+'.png', dpi=800)
    return model_name_2, model_corr, num_displace_greater10, num_displace_lesser5, rank_diff

def main():
    plt.figure(figsize=(14, 12))
    plt.title('Model Comparisons')
    plt.axis('off')
    models = [C.ALL_MPNET_BASE_V2, C.ALL_DISTILROBERTA_V1, C.MINILM_L12_V2, C.PARAPHRASE_ALBERT_SMALL_V2]
    comparisons_df = pd.DataFrame(columns=['Model', 'Correlation', 'PValue', 'Rank Diplace10+', 'Rank Displace5-', 'rank_diff'])
    for i in range(4):
        model_name_2, model_corr, num_displaced_greater10, num_displaced_lesser5, rank_diff = compare_models(C.FILE_NAME, C.MINILM_L6_V2, models[i], "Haodong Tao", i+1)
        comparisons_df.loc[i] = [model_name_2, model_corr.statistic, model_corr.pvalue, num_displaced_greater10, num_displaced_lesser5, rank_diff]
    plt.savefig('model_comparisons.png', dpi=800)

    plt.figure(figsize=(14, 7))
    plt.title('Model Rank Comparisons with all-MiniLM-L6-v2')
    comparisons_df[['Model', 'Correlation', 'PValue', 'Rank Diplace10+', 'Rank Displace5-']].to_csv('Model_Comparisons_with_MINILM_L6_V2.csv', index=False, encoding='utf-8')
    box_plot_df = pd.DataFrame(columns=comparisons_df['Model'].str.split('/').str[1])
    box_plot_df['all-mpnet-base-v2'] = comparisons_df.iloc[0, -1]
    box_plot_df['all-distilroberta-v1'] = comparisons_df.iloc[1, -1]
    box_plot_df['all-MiniLM-L12-v2'] = comparisons_df.iloc[2, -1]
    box_plot_df['paraphrase-albert-small-v2'] = comparisons_df.iloc[3, -1]
    box_plot = box_plot_df.boxplot(column=list(box_plot_df.columns))
    box_plot.plot()
    plt.savefig('model_rank_comparisons_with_all-MiniLM-L6-v2.png', dpi=800)

if __name__ == "__main__":
    main()

# # draw graph
# plt.figure(figsize=(20, 10))

# plt.subplot(2, 1, 1)
# plt.scatter(x, cosine_diff, s=8)
# plt.axhline(y=0, color="red", linestyle="-")
# plt.tick_params(axis="x", bottom=False, labelbottom=False)
# plt.ylabel("Cosine Difference")
# for i in range(len(top_matches_1)):
#     plt.annotate(top_matches_1[i][0][:6], (x[i], cosine_diff[i]), fontsize="8")

# plt.subplot(2, 1, 2)
# plt.scatter(x, rank_diff, s=8)
# plt.axhline(y=0, color="red")
# plt.tick_params(axis="x", bottom=False, labelbottom=False)
# plt.ylabel("Ranks gained")
# for i in range(len(top_matches_1)):
#     plt.annotate(top_matches_1[i][0][:6], (x[i], rank_diff[i]), fontsize="8")

# plt.suptitle(
#     f"Difference in cosine similarity and ranking of everyone vs {target_person} between {model_name_1} and {model_name_2}",
#     size=16,
# )
# plt.show()





# def main():
#     compare_models(C.FILE_NAME, C.MINILM_L6_V2, C.ALL_MPNET_BASE_V2, "Greg Kirczenow")


# if __name__ == "__main__":
#     main()