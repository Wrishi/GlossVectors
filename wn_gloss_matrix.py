import numpy as np
import scipy
from scipy.sparse import bsr_matrix
import nltk.corpus
from nltk.tokenize import RegexpTokenizer
import re
#import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn

def create_gloss_matrix(neighbors = 4, min_freq = 4, path = "", gloss_filename = 'gloss_matrix.npz', word_filename = 'wn_words.npz'):
    num_pos = neighbors
    wn_corpus = ""
    
    # Creating corpus
    print("Creating corpus...")
    wn_corpus = ""
    #for ss in wn.all_synsets():
    #    wn_corpus += ss.definition() + ". " + ". ".join(ss.examples()) + ". "
    covered = []
    for ss in wn.all_synsets():
        w = ss.name().split(".")[0]
        if w not in covered:
            syns = wn.synsets(w)
            for s in syns:
                wn_corpus += s.definition() + ". " + ". ".join(s.examples()) + ". "
            covered.append(w)
    # Corpus to array
    wn_words = re.sub("[^\w]", " ",  wn_corpus).split()
    
    # Removing Stopwords
    print("Removing Stopwords...")
    wn_filtered_words = [word for word in wn_words if word not in nltk.corpus.stopwords.words('english')]
    
    # Removing low frequency words
    ### Creating dictionary for frequency of words
    print("Removing low frequency words...")
    wn_dict = {}
    for w in wn_filtered_words:
        word = w.lower()
        if word not in wn_dict.keys():
            wn_dict[word] = 0
        
        wn_dict[word] += 1
        
    wn_dict_copy = wn_dict.copy()
    
    ### Number of words for which frequency is less than num are chucked out
    ### [Selected 4 since it give the size that matches with paper]
    num = min_freq
    c = 0
    for w, n in wn_dict_copy.items():
        if n < num:
            del wn_dict[w]
            c += 1
            
    wn_unique_words = list(wn_dict.keys())
    
    # Creating gloss matrix for all words in filtered corpus
    print("Creating gloss matrix. This will take some time...")
    wn_gloss_matrix = bsr_matrix((len(wn_unique_words), len(wn_unique_words)), dtype=np.int8).toarray()
    r = 0
    for w in wn_unique_words:
        wn_words_around = []
        indices = [i for i, x in enumerate(wn_filtered_words) if x == w]
        for i in indices:
            wn_words_around.extend(wn_filtered_words[i - num_pos : i])
            wn_words_around.extend(wn_filtered_words[i + 1 : i + num_pos + 1])
        
        for word in wn_words_around:
            if word in wn_unique_words:
                index = wn_unique_words.index(word)
                wn_gloss_matrix[r][index] += 1
        
        r += 1
        
    np.save(path + gloss_filename, wn_gloss_matrix)
    print("Gloss Matrix was saved in file: ", path + gloss_filename)
        
    np.save(path + word_filename, wn_unique_words)
    print("Word list was saved in file: ", path + word_filename)
    print("Position of words correspond to Vectors in matrix.")
