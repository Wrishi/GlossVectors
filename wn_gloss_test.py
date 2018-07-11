import numpy as np
import wn_gloss_matrix as wgm

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

wgm.create_gloss_matrix(neighbors = 5)

wn_unique_words = np.load("wn_words.npz.npy")
wn_gloss_matrix = np.load("gloss_matrix.npz.npy")

for w1, w2 in test_words:
    try:
        i1 = np.where(wn_unique_words==w1)
        i2 = np.where(wn_unique_words==w2)
        v1 = wn_gloss_matrix[i1][0]
        v2 = wn_gloss_matrix[i2][0]
        
        similarity = np.dot(v1, v2)/(np.linalg.norm(v1)* np.linalg.norm(v2))
        print(w1, ", ", w2, ": ", similarity)
    except IndexError:
        print("Word not found.")
        continue
