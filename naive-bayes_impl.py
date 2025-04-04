from datasets import load_dataset
import math
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def compute_word_probabilities(dataset):
    """
    Computes conditional probabilities P(word|class) for each word and class.
    This is the "training" phase of Naive Bayes.
    """
    # Initialize counts with basic dictionaries
    word_counts = {}  # Will store the count of each word in each class
    class_counts = {}  # Will store the total count of documents in each class

    # Count words and classes
    for example in dataset:
        text = example['text']
        label = example['label']
        # Tokenize and convert to lowercase to standardize
        words = word_tokenize(text.lower())
        
        # Increment the count of documents for this class
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
        
        # Initialize word_counts dictionary for this label if needed
        if label not in word_counts:
            word_counts[label] = {}
            
        # Count occurrences of each word in this document for this label
        for word in words:
            if word not in word_counts[label]:
                word_counts[label][word] = 0
            word_counts[label][word] += 1

    # Calculate probabilities using Laplace (add-1) smoothing
    total_words_per_class = {}
    for label in word_counts:
        # Count total words in each class for denominator
        total_words_per_class[label] = sum(word_counts[label].values())
    
    # Calculate P(word|class) = (count(word,class) + 1) / (count(all words in class) + |V|)
    # where |V| is vocabulary size (Laplace smoothing)
    word_probabilities = {}
    for label in word_counts:
        word_probabilities[label] = {}
        vocab_size = len(word_counts[label])
        total_count = total_words_per_class[label]
        
        for word, count in word_counts[label].items():
            # Apply Laplace smoothing to avoid zero probabilities
            word_probabilities[label][word] = (count + 1) / (total_count + vocab_size)

    return word_probabilities, class_counts

def predict(text, word_probabilities, class_counts):
    """
    Predicts the class of a text using Naive Bayes algorithm.
    Implements P(class|text) ∝ P(class) × ∏ P(word|class)
    Using log probabilities to prevent underflow.
    """
    words = word_tokenize(text.lower())
    class_scores = {}
    total_examples = sum(class_counts.values())

    # Calculate log probabilities for each class
    for label in class_counts:
        # Start with the prior probability P(class)
        class_scores[label] = math.log(class_counts[label] / total_examples)

        # Add log of conditional probabilities P(word|class) for each word
        for word in words:
            if word in word_probabilities[label]:
                # Using log probabilities to avoid numerical underflow
                class_scores[label] += math.log(word_probabilities[label][word])
            # Note: Words not in the vocabulary are implicitly ignored

    # Convert log probabilities to actual probabilities and normalize
    # exp(log(p)) = p
    total_score = sum(math.exp(score) for score in class_scores.values())
    normalized_scores = {}
    for label, score in class_scores.items():
        # Normalize to get probabilities that sum to 1
        normalized_scores[label] = math.exp(score) / total_score

    # Returns a dictionary mapping each class to its probability
    return normalized_scores

if __name__ == "__main__":
    # Load the IMDB dataset (contains movie reviews labeled as positive or negative)
    dataset = load_dataset('imdb', split='train')

    # Train the Naive Bayes model by computing word probabilities
    word_probabilities, class_counts = compute_word_probabilities(dataset)

    # Example prediction on a short text
    example_text = "This movie was really bad! I hated it."
    predictions = predict(example_text, word_probabilities, class_counts)
    
    # Output will show probability for each class (0=negative, 1=positive)
    print("Predictions:", predictions)
    
    # Convert numeric labels to readable format
    label_names = {0: "negative", 1: "positive"}
    readable_predictions = {label_names[label]: prob for label, prob in predictions.items()}
    print("Sentiment probabilities:", readable_predictions)