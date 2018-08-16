import numpy as np
import wn_gloss_matrix as wgm
import math
from scipy import spatial
from scipy.stats import spearmanr
from nltk.corpus import wordnet as wn

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

path = "output/"
g_fn = 'gloss_matrix.clean_text.v04'
w_fn = 'wn_words.clean_text.v04'

# Takes long time. To be used only when it is needed to create gloss matrix
# wgm.create(neighbors = 5, gloss_filename = g_fn, word_filename = w_fn)
# wgm.create(corpus_file = "wn_corpus.txt", neighbors = 5, gloss_filename = g_fn, word_filename = w_fn)

wn_gloss_matrix = np.load(path+g_fn+".npy")
wn_unique_words = np.load(path+w_fn+".npy")

"""
print("Angle between reference word vectors (paper): ")
for w1, w2 in paper_words:
    try:
        i1 = np.where(wn_unique_words==w1)
        i2 = np.where(wn_unique_words==w2)
        v1 = wn_gloss_matrix[i1][0]
        v2 = wn_gloss_matrix[i2][0]
        
        print(w1, "<->", w2, "=", spearmanr(v1, v2)[0])
        #print("cosine similarity: ", 1 - spatial.distance.cosine(v1,v2))
        #print("euclidean distance: ", np.linalg.norm(v1-v2))
    except IndexError:
        print("Word not found.")
        continue

print('=====================================')
print("\n\nAngle between test word vectors: ")
for w1, w2 in test_words:
    try:
        
        i1 = np.where(wn_unique_words==w1)
        i2 = np.where(wn_unique_words==w2)
        v1 = wn_gloss_matrix[i1][0]
        v2 = wn_gloss_matrix[i2][0]
        
        print(w1, "<->", w2, "=", spearmanr(v1, v2))
        #print("cosine similarity: ", 1 - spatial.distance.cosine(v1,v2))
        #print("euclidean distance: ", np.linalg.norm(v1-v2))
    except IndexError:
        print("Word not found.")
        continue

"""
print('=====================================')
print("\n\nAngle between test word vectors with hyponyms and hypernyms: ")
similarities = []
for w1, w2 in test_words:
    try:
        w1_ss = wn.synset(w1+".n.01")
        w2_ss = wn.synset(w2+".n.01")
        
        #print(w1_ss)
        #print(w2_ss)
        
        w1_list = w1_ss.hypernyms()
        w2_list = w2_ss.hypernyms()
        w1_list.extend(w1_ss.hyponyms())
        w2_list.extend(w2_ss.hyponyms())
        
        #print(w1_list)
        #print(w2_list)
        
        i1 = np.where(wn_unique_words==w1)
        i2 = np.where(wn_unique_words==w2)
        v1 = wn_gloss_matrix[i1][0]
        v2 = wn_gloss_matrix[i2][0]
        
        v1s = [v1]
        v2s = [v2]
        
        #print(v1s)
        #print(v2s)
        
        for h in w1_list:
            for l in h.lemmas():
                w = l.name().split(".")[0]
                i = np.where(wn_unique_words==w)
                if i[0].size == 0:
                    continue
                v = wn_gloss_matrix[i][0]
                v1s.append(v)
                
        for h in w2_list:
            for l in h.lemmas():
                w = l.name().split(".")[0]
                i = np.where(wn_unique_words==w)
                if i[0].size == 0:
                    continue
                v = wn_gloss_matrix[i][0]
                v2s.append(v)
        
        values = []
        for v1 in v1s:
            for v2 in v2s:
                values.append(spearmanr(v1, v2))
        values = np.array(values)
        
        #print(w1, "<->", w2, "=", "AVG:", np.mean(values), ", MAX:", np.amax(values))
        
        similarities.append(np.amax(values))
        print(w1, "<->", w2, "=", np.amax(values))   
    except IndexError:
        print("Word not found.")
        similarities.append(0.0)
        continue

print("\nSimilarity Array (0.0 for words not found):")
print(similarities)
