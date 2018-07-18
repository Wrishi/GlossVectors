import numpy as np
import wn_gloss_matrix as wgm
import math

paper_words = [
    ('serve', 'tennis'),
    ('serve', 'food'),
    ('fork', 'tennis'),
    ('fork', 'food'),
    ('player', 'tennis'),
    ('player', 'food'),
]

test_words = [
    ('car', 'automobile'),
    ('gem', 'jewel'),
    ('journey', 'voyage'),
    ('boy', 'lad'),
    ('coast', 'shore'),
    ('asylum', 'madhouse'),
    ('magician', 'wizard'),
    ('midday', 'noon'),
    ('furnace', 'stove'),
    ('food', 'fruit'),
    ('bird', 'cock'),
    ('bird', 'crane'),
    ('tool', 'implement'),
    ('brother', 'monk'),
    ('lad', 'brother'),
    ('crane', 'implement'),
    ('journey', 'car'),
    ('monk', 'oracle'),
    ('cemetery', 'woodland'),
    ('food', 'rooster'),
    ('coast', 'hill'),
    ('forest', 'graveyard'),
    ('shore', 'woodland'),
    ('monk', 'slave'),
    ('coast', 'forest'),
    ('lad', 'wizard'),
    ('chord', 'smile'),
    ('glass', 'magician'),
    ('rooster', 'voyage'),
    ('noon', 'string')
]

g_fn = 'gloss_matrix.clean_text'
w_fn = 'wn_words.clean_text'

# Takes long time. To be used only when it is needed to create gloss matrix
# wgm.create(neighbors = 5, gloss_filename = g_fn, word_filename = w_fn)
wgm.create(corpus_file = "wn_corpus.txt", neighbors = 5, gloss_filename = g_fn, word_filename = w_fn)

wn_gloss_matrix = np.load(g_fn+".npy")
wn_unique_words = np.load(w_fn+".npy")


print("Angle between reference word vectors (paper): ")
for w1, w2 in paper_words:
    try:
        print('=====================================')
        i1 = np.where(wn_unique_words==w1)
        i2 = np.where(wn_unique_words==w2)
        v1 = wn_gloss_matrix[i1][0]
        v2 = wn_gloss_matrix[i2][0]
        
        d = np.dot(v1, v2)/(np.linalg.norm(v1)* np.linalg.norm(v2))
        print(w1, ", ", w2)
        print(math.acos(d) * 180/math.pi)
        print("cosine distance: ", d)
        print("euclidean distance: ", np.linalg.norm(v1-v2))
    except IndexError:
        print("Word not found.")
        continue


print("\nAngle between test word vectors: ")
for w1, w2 in test_words:
    try:
        print('=====================================')
        i1 = np.where(wn_unique_words==w1)
        i2 = np.where(wn_unique_words==w2)
        v1 = wn_gloss_matrix[i1][0]
        v2 = wn_gloss_matrix[i2][0]
        
        d = np.dot(v1, v2)/(np.linalg.norm(v1)* np.linalg.norm(v2))
        print(w1, ", ", w2)
        print(math.acos(d) * 180/math.pi)
        print("cosine distance: ", d)
        print("euclidean distance: ", np.linalg.norm(v1-v2))
    except IndexError:
        print("Word not found.")
        continue
