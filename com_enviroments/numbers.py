import os
import numpy as np
import requests
import re
from ast import literal_eval
import matplotlib.pyplot as plt
from com_enviroments.BaseEnviroment import BaseEnviroment
import torchHelpers as th
import torch.nn.functional as F
from collections import Counter
import torch

class NumberEnvironment(BaseEnviroment):
    def __init__(self, prior='ngram') -> None:
        super().__init__()
        self.prior = prior
        if self.prior == 'ngram':
            self.num_use_dist = self.get_use_dist()
            # self.num_use_dist = np.ones(3) /3
            self.data_dim = 100
        elif self.prior == 'MSCoco20':
            self.num_use_dist = np.load('data/MSCoco_1-20.npy')
            self.data_dim = 20

        self.samp = 'freq'
        self.numbers = np.array(list(range( self.data_dim)))
       # plt.plot(range(1, 101), self.num_use_dist)
       # plt.savefig('fig/wrd_dist.png')
    def full_batch(self):
        return self.numbers, np.expand_dims(self.numbers, axis=1)

    def mini_batch(self, batch_size=10):
        if self.samp == 'freq':
            # Sample from categorical
            batch = np.expand_dims(np.random.choice(a=self.numbers, size=batch_size, replace=True, p=self.num_use_dist), axis=1)
        else:
            batch = np.expand_dims(np.random.randint(0, self.data_dim, batch_size), axis=1)
        return batch, batch + 1

    def sim_index(self, num_a, num_b):
        return self.data_dim - np.sqrt(np.power(num_a-num_b, 2))

    def get_use_dist(self):
        fname = 'data/num_use_dist.npy'
        if os.path.isfile(fname):
            data = np.load(fname)
        else:
            data = []
            for i in range(10):
                numbers = range(i*10 + 1,(i+1)*10 + 1)
                query = ','.join([str(n) for n in numbers])
                params = dict(content=query, year_start=1999, year_end=2000,
                              corpus=15, smoothing=3)
                import requests
                req = requests.get('http://books.google.com/ngrams/graph', params=params)
                res = re.findall('var data = (.*?);\\n', req.text)
                data += [qry['timeseries'][1] for qry in literal_eval(res[0])]
            data = np.array(data)
            data /= data.sum()
            print(data)
            np.save(fname, data)
        return data

    def number_reward(self, target, guess):
        # Distance Reward
        diff = torch.abs(target - guess.unsqueeze(dim=1))
        reward = 1-(diff.float()/100)
        # Importance Reward
        delta = 1
        eps = 0.1
        mask = torch.abs(target - guess.unsqueeze(dim=1)) == 0
        reward[mask] = reward[mask] + delta * 1 /((eps+target[mask].float()))
        return reward

    def inverted_number(self, target, guess):
        return (torch.abs(target - guess.unsqueeze(dim=1)) == 0) * 1/(0.001+target.float())

    def interval_reward(self, target, guess):
        c = 0.001
        diff = torch.abs(target - guess.unsqueeze(dim=1))
        reward = (target<5)*100*2**(-diff*c) + 10*2**(-diff*c)
        return reward

    def target_reward(self, target, guess):
        diff = torch.abs(target - guess.unsqueeze(dim=1))
        return (diff==0).float()*1/((1+target.float()))

    def word2number(self, agent):
        msg = th.float_var(np.eye(agent.msg_dim))
        guess_logits = agent(msg=msg)
        guess_probs = F.softmax(guess_logits, dim=1)
        _, guess = guess_probs.max(1)
        guess = guess.data.numpy()
        dublicates = [item for item, count in Counter(guess).items() if count > 1]
        return -len(dublicates) * 2
