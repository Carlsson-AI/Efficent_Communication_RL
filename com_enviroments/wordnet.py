import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from ast import literal_eval
import pandas import pd
import numpy as np
import re
import requests
import subprocess
import sys

import torch
import torchHelpers as th
from com_enviroments.BaseEnviroment import BaseEnviroment
from getngrams import runLargeQuery


class WordNet_Environment(BaseEnviroment):
    def __init__(self, wn_path='data/wordnet/', dist='ngram') -> None:
        super().__init__()
        self.all_lemmas = wn.all_lemma_names()
        self.all_synsets = wn.all_synsets()
        self.data_dim = len(all_words)
        self.dist = self.get_dist(dist)


    def get_dist()
