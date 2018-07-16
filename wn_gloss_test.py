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

g_fn = 'gloss_matrix.v01.npz'
w_fn = 'wn_words.v01.npz'

# Takes long time. To be used only when it is needed to create gloss matrix
wgm.create_gloss_matrix(neighbors = 5, gloss_filename = g_fn, word_filename = w_fn)

wn_unique_words = np.load(g_fn)
wn_gloss_matrix = np.load(w_fn)

print("Angle between reference word vectors (paper): ")
for w1, w2 in paper_words:
    try:
        i1 = np.where(wn_unique_words==w1)
        i2 = np.where(wn_unique_words==w2)
        v1 = wn_gloss_matrix[i1][0]
        v2 = wn_gloss_matrix[i2][0]
        
        similarity = math.acos(np.dot(v1, v2)/(np.linalg.norm(v1)* np.linalg.norm(v2))) * 180/math.pi
        print(w1, ", ", w2, ": ", similarity)
    except IndexError:
        print("Word not found.")
        continue


print("\nAngle between test word vectors: ")
for w1, w2 in test_words:
    try:
        i1 = np.where(wn_unique_words==w1)
        i2 = np.where(wn_unique_words==w2)
        v1 = wn_gloss_matrix[i1][0]
        v2 = wn_gloss_matrix[i2][0]
        
        similarity = math.acos(np.dot(v1, v2)/(np.linalg.norm(v1)* np.linalg.norm(v2))) * 180/math.pi
        print(w1, ", ", w2, ": ", similarity)
    except IndexError:
        print("Word not found.")
        continue
