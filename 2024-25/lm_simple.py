from transformers import BertTokenizer
from collections import defaultdict
import nltk

# Inizializzare il tokenizer BERT per l'italiano
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-italian-uncased')

def count_subsequent(corpus, prefix):
    counter = defaultdict(lambda: 0)
    for doc in corpus:
        if doc[:len(prefix)] == prefix:
            counter[doc[len(prefix)]] += 1
    return dict(counter)

def count_markov(corpus, prefix, k):
    counter = defaultdict(lambda: 0)
    for doc in corpus:
        for gram in nltk.ngrams(doc, n=k+1):
            if list(gram[:-1]) == prefix:
                counter[gram[-1]] += 1
    return dict(counter)

# Frasi con i marcatori [#S] e [#E]
sentences = [
    "Il sole splende forte, il sole splende caldo.",
    "La pioggia cade leggera, la pioggia bagna le strade.",
    "Il cane corre veloce, il cane salta alto.",
    "La macchina è veloce sulla strada mentre la pioggia cade fredda.",
    "Il sole splende e il gatto si sveglia presto.",
    "La pioggia cade leggera, la pioggia cade tutto il giorno.",
    "Il sole è alto e il mare è calmo.",
    "Il libro è interessante, il libro racconta una storia lunga.",
    "Il libro è bello e il bambino ride felice.",
    "La scuola è chiusa oggi, la scuola riapre domani."
]
corpus = []
for sentence in sentences:
    tokens = ["[#S]"] + tokenizer.tokenize(sentence) + ["[#E]"]
    corpus.append(tokens)
    
for doc in corpus:
    print(" ".join(doc))

print(count_subsequent(corpus=corpus, prefix=["[#S]", "la", "pioggia", "cade"]))
print(count_markov(corpus=corpus, prefix=["cade", "fredda"], k=2))
