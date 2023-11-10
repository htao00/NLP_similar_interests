import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def compare_embedding(s1, s2):
    """
    Given two strings, compare their embeddings' cosine similarity.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    e1 = model.encode(s1)
    e2 = model.encode(s2)

    res = cosine_similarity(e1, e2)
    return res


data = pd.read_csv('data.csv')

def calc_cosine_sims():
    ### modify sentences
    original = "I like hiking, swimming, traveling, enjoying nice weather and sun (and h.a.t.e winters!)"

    # Change word order
    # hiking, swimming, traveling -> swimming, traveling, hiking
    mod_1 = "I like swimming, traveling, hiking, enjoying nice weather and sun (and h.a.t.e winters!)"
    # move "swimming, traveling, hiking" to the end
    mod_2 = "I like enjoying nice weather and sun (and h.a.t.e winters!), swimming, traveling, hiking"

    # Substitute with synonyms
    # h.a.t.e -> hate
    mod_3 = "I like hiking, swimming, traveling, enjoying nice weather and sun (and hate winters!)"
    # like -> love
    mod_4 = "I love hiking, swimming, traveling, enjoying nice weather and sun (and h.a.t.e winters!)"
    # like -> luv
    mod_5 = "I luv hiking, swimming, traveling, enjoying nice weather and sun (and h.a.t.e winters!)"

    # Use antonyms
    # like -> don't like
    mod_6 = "I don't like hiking, swimming, traveling, enjoying nice weather and sun (and h.a.t.e winters!)"
    # like -> don't like; h.a.t.e -> love
    mod_7 = "I don't like hiking, swimming, traveling, enjoying nice weather and sun (and love winters!)"
    # like -> hate
    mod_8 = "I hate hiking, swimming, traveling, enjoying nice weather and sun (and h.a.t.e winters!)"
    # like -> hate; h.a.t.e -> love
    mod_9 = "I hate hiking, swimming, traveling, enjoying nice weather and sun (and love winters!)"

    # Shorten sentence
    mod_10 = "I like hiking, swimming, traveling, enjoying nice weather and sun"
    mod_11 = "I like hiking, swimming, traveling and h.a.t.e winters"

    # Rewrite with opposite meaning
    mod_12 = "I like sitting at home, enjoy cold weather and winters"
    mod_13 = "I like sitting at home"
    print("Cosine similarity of the original sentence vs modified sentences")

    print("\nChange word order")
    print(mod_1, compare_embedding([original], [mod_1]))
    print(mod_2, compare_embedding([original], [mod_2]))

    print("\nSubstitute with synonyms")
    print(mod_3, compare_embedding([original], [mod_3]))
    print(mod_4, compare_embedding([original], [mod_4]))
    print(mod_5, compare_embedding([original], [mod_5]))

    print("\nUse antonyms")
    print(mod_6, compare_embedding([original], [mod_6]))
    print(mod_7, compare_embedding([original], [mod_7]))
    print(mod_8, compare_embedding([original], [mod_8]))
    print(mod_9, compare_embedding([original], [mod_9]))

    print("\nShorten sentence")
    print(mod_10, compare_embedding([original], [mod_10]))
    print(mod_11, compare_embedding([original], [mod_11]))

    print("\nRewrite with opposite meaning")
    print(mod_12, compare_embedding([original], [mod_12]))
    print(mod_13, compare_embedding([original], [mod_13]))

def cossim_for_unrelated_sentences():
    original_greg = 'I enjoy being outdoors in nature, trail running and surfing.'
    mod_greg = "She finally understood that grief was her love with no place for it to go."
    original_tao = 'I regularly exercise at the gym, enjoy hanging out with friends, and cooking and eating delicious food'
    mod_tao = 'Waffles are always better without fire ants and fleas.'
    print('comparing:\n"',original_greg, '" with: \n"', mod_greg, '"')
    print(compare_embedding([original_greg], [mod_greg]))
    print('comparing:\n"',original_tao, '" with: \n"', mod_tao, '"')
    print(compare_embedding([original_tao], [mod_tao]))


calc_cosine_sims()
cossim_for_unrelated_sentences()