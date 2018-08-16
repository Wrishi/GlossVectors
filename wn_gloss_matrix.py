import numpy as np
import scipy
from scipy.sparse import bsr_matrix
import nltk.corpus
from nltk.tokenize import RegexpTokenizer
import re
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import os


def create_corpus():
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
            
    return wn_corpus


def save_corpus(wn_corpus, corpus_file = "wn_corpus.txt"):
    # Saving corpus
    print("Saving corpus...")
    text_file = open(corpus_file, "w")
    text_file.write(wn_corpus)
    text_file.close()
    print("Corpus saved in file: ", corpus_file)
    

def load_corpus(corpus_file):
    print("Loading corpus...")
    return open(corpus_file, "r").read()

"""
def get_clean_word_list_old(wn_corpus):
    # Corpus to array
    wn_words = re.sub("[^\w]", " ",  wn_corpus).split()
    
    # Removing Stopwords
    print("Removing Stopwords...")
    wn_filtered_words = [word for word in wn_words if word not in nltk.corpus.stopwords.words('english')]
    
    return wn_filtered_words
"""

def get_clean_word_list(wn_corpus):
    print("Cleaning corpus...")
    tokens = word_tokenize(wn_corpus)
    # convert to lower case
    tokens = [w.lower() for w in tokens]

    # keep only nouns <========= A bad idea probably
    # pos_tagged = nltk.pos_tag(tokens)
    # nouns = [word for (word, pos) in pos_tagged if pos in ['NN','NNP','NNS','NNPS']]

    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    
    # remove words with 2 alphabets or less
    words = [word for word in words if len(word) > 1]
    
    # filter out stop words
    # discarding nltk stopword list to use the one used in paper
    # stop_words = set(stopwords.words('english')) 
    stop_words = open('stoplist.txt', 'r').read()
    stop_words = stop_words.split('\n')
    
    wn_filtered_words = [w for w in words if not w in stop_words]
    
    return wn_filtered_words


def get_filtered_word_list(wn_filtered_words, min_freq = 4):
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
    
    return wn_unique_words, wn_dict


def create_gloss_matrix(wn_filtered_words, wn_unique_words, neighbors):
    num_pos = neighbors
    
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
        
    return wn_gloss_matrix


def save_gloss_matrix(wn_gloss_matrix, gloss_filename, path = ""):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + gloss_filename, wn_gloss_matrix)
    print("Gloss Matrix was saved in file: ", path + gloss_filename)


def save_unique_words(wn_unique_words, word_filename, path = ""):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + word_filename, wn_unique_words)
    print("Word list was saved in file: ", path + word_filename)
    print("Position of words correspond to Vectors in matrix.")


def create(corpus_file = None,
           neighbors = 4, 
           min_freq = 4,
           output_path = "output/", 
           gloss_filename = "gloss_matrix", 
           word_filename = "wn_words"):
    if corpus_file == None:
        wn_corpus = create_corpus()
        save_corpus(wn_corpus)
    else:
        wn_corpus = load_corpus(corpus_file)
        
    wn_filtered_words = get_clean_word_list(wn_corpus)
    wn_unique_words, wn_dict = get_filtered_word_list(wn_filtered_words, min_freq)
    wn_gloss_matrix = create_gloss_matrix(wn_filtered_words, wn_unique_words, neighbors)
    
    save_gloss_matrix(wn_gloss_matrix, gloss_filename, output_path)
    save_unique_words(wn_unique_words, word_filename, output_path)

        
    
