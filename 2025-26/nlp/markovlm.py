import nltk
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from collections import defaultdict
from typing import List
from tqdm import tqdm
from nltk.tokenize import sent_tokenize


class MarkovLM:
    """Implements a Markov LM
    """
    def __init__(self, k: int = 2, tokenizer_model: str = "dbmdz/bert-base-italian-uncased"):
        self.k = k 
        self.unigram = defaultdict(lambda: 1)
        self.k_index = defaultdict(lambda: defaultdict(lambda: 1))
        self.U = float('inf')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
        self.start_symbol = "[#S]"
        self.end_symbol = "[#E]"
        
    def train(self, corpus: List[str]):
        """fill if the indexes

        Args:
            corpus (List[str]): List of textual documents
        """
        for document in tqdm(corpus):
            try:
                tokens = self.tokenizer.tokenize(document)
                for keys in nltk.ngrams(tokens, n=self.k, pad_left=True, 
                                        pad_right=True, 
                                        left_pad_symbol=self.start_symbol, 
                                        right_pad_symbol=self.end_symbol):
                    self.k_index[keys[:-1]][keys[-1]] += 1
                    for k in keys:
                        self.unigram[k] += 1
            except TypeError:
                pass
    
    def pickup(self, prefix: tuple = None):
        if prefix is None:
            # unigram
            s = pd.Series(self.unigram) / sum(self.unigram.values())
            return np.random.choice(s.index.values, p=s.values)
        else:
            assert len(prefix) == self.k - 1
            data = self.k_index[prefix]
            s = pd.Series(data)
            if s.empty:
                token = self.pickup()
            else:
                s = s / s.sum()
                token = np.random.choice(s.index.values, p=s.values)
            return token
    
    def generate(self, prefix: tuple = None, unigram: bool = False, max_len: int = 2000):
        text = []
        if prefix is None:
            prefix = tuple([self.start_symbol] * (self.k - 1))
        text.extend(prefix)
        for i in range(max_len):
            if unigram:
                token = self.pickup()
            else:
                token = self.pickup(prefix=prefix)
            text.append(token)
            if token == self.end_symbol:
                break
            else:
                prefix = tuple(text[-(self.k - 1):])
        return text
        
    @staticmethod
    def read_txt(file_path: str):
        with open(file_path, 'r') as infile:
            text = infile.read()
        return sent_tokenize(text)