from datasets import load_dataset
from collections import Counter, defaultdict
import math
from nltk.tokenize import word_tokenize, sent_tokenize

#dataset with texts
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

def ngram_probability():
    return