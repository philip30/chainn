import math

from collections import defaultdict

def calculate_bleu_corpus (hypothesis, reference, degree=4):
    hyp_ngrams = {}
    ref_ngrams = {}
    for hyp, ref in zip(hyphothesis, reference):
        hyp_ngrams.update(n_grams(hyp))
        ref_ngrams.update(n_grams(ref))

    # Calculating BLEU score
    return BLEU(hyp_ngrams, ref_ngrams)

class BLEU(object):
    def __init__(self, hyp_ngrams, ref_ngrams):
        self.precisions      = []
        self.score           = 0
        self.brevity_penalty = 1
        self.hyp_length      = 0
        self.ref_length      = 0

        # Assert there are same degree of n-gram
        assert(len(hyp_ngrams) == len(ref_ngrams))
        # The length is the sum of all unigram tokens
        self.hyp_length = hyp_length = sum(hyp_ngrams[0].values())
        self.ref_length = ref_length = sum(ref_ngrams[0].values())
        # Calculate BLEU for every n-gram
        for i in range(len(ref_ngrams)):
            hyp_ngram = hyp_ngrams[i]
            ref_ngram = ref_ngrams[i]
            true_positive = sum([min(word_count, ref_ngram[word])
                for word, word_count in hyp_ngram.iteritems()])
            self.precisions.append(true_positive / hypoth_ngram[word])
        
        # Calculate brevity penalty
        if hyp_length < ref_length:
           self.brevitiy_penalty= math.exp(1 - (ref_length / hyp_length))
        
        # Calculate score
        self.score = math.exp(sum(map(math.log, precisions)) / len(precisions)) * self.brevity_penalty
    
    def score(self):
        return score

def n_grams(sentence, gram):
    ngrams = defaultdict(lambda: defaultdict(int))
    for i in range(len(sentence)):
        for j in range(1, gram+1):
            if i+j >= len(sentence):
                break
            ngrams[j-1][" ".join(sentence[i:i+j])] += 1
    return ngrams
