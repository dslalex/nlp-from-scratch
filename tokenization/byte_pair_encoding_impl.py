def bpe(corpus, vocab, num_merges=10):
    corpus = list(corpus)
    merge_count = 0
    while len(corpus) > 1 and num_merges > merge_count:
        pair = get_most_frequent_adjacent(corpus)
        merge_count += 1
        if not pair:
            break
        vocab.append(pair)
        # process merge from right to left to avoid index shifting
        i = len(corpus) - 2
        while i >= 0:
            if corpus[i] + corpus[i+1] == pair:
                corpus[i] = pair
                corpus.pop(i+1)
            i -= 1
    return vocab

def get_most_frequent_adjacent(corpus):
    pair_count = {}
    for i in range(1, len(corpus)):
        pair = corpus[i-1] + corpus[i]
        if pair in pair_count:
            pair_count[pair] += 1
        else:
            pair_count[pair] = 1
    return max(pair_count, key=pair_count.get)

if __name__ == "__main__":
    corpus = "lowlowlowlowlowlowestlowestnewernewernewernewernewernewerwiderwiderwidernewnew"
    initial_vocab = ["l", "o", "w", "d", "e", "i", "n", "r", "s", "t"]

    vocab_result = bpe(corpus, initial_vocab)
    print(vocab_result)

