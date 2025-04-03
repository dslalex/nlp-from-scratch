from datasets import load_dataset
from collections import Counter, defaultdict
import math
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize

#dataset with texts
dataset = load_dataset('imdb', split='train[:10%]')

def ngram_probability(dataset, n=3):
    # Tokenize the text into sentences
    sentences = []
    for text in dataset['text']:
        sentences.extend(sent_tokenize(text))

    # Tokenize the sentences into words
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

    ngrams = []
    for sentence in tokenized_sentences:
        for i in range(len(sentence) - n + 1):
            ngrams.append(tuple(sentence[i:i+n]))

    # Count the frequency of each n-gram
    ngram_counts = Counter(ngrams)

    # Calculate probabilities
    total_ngrams = sum(ngram_counts.values())
    ngram_probabilities = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}

    return ngram_probabilities

# Example usage
if __name__ == "__main__":
    ngram_probs = ngram_probability(dataset, 2)
    # Print the top 10 most common n-grams and their probabilities
    for ngram, prob in list(ngram_probs.items())[:10]:
        print(f"N-gram: {ngram}, Probability: {prob:.10f}")