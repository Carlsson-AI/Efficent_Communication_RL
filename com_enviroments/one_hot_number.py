import os
import numpy as np
import requests
import re
from ast import literal_eval
import matplotlib.pyplot as plt
from com_enviroments.BaseEnviroment import BaseEnviroment
import torch

class OneHotNumberEnvironment(BaseEnviroment):
    def __init__(self, data_dim):
        super().__init__()
        self.data_dim = data_dim
        self.samp = 'freq'
        self.numbers = np.array(list(range( self.data_dim)))
        self.num_use_dist = self.get_use_dist()
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

        onehot = torch.FloatTensor(len(batch), self.data_dim)
        onehot.zero_()
        batch = Variable(onehot.scatter_(1, batch.data.unsqueeze(1), 1))
        return batch, batch

    def sim_index(self, num_a, num_b):
        return self.data_dim - np.sqrt(np.power(num_a-num_b, 2))

    def get_use_dist(self):
        fname = 'data/num_use_dist.npy'
        if os.path.isfile(fname):
            data = np.load(fname)
        else:
            data = []
            for i in range(10):
                # [1,100]
                numbers = range(i*10 + 1,(i+1)*10 + 1)
                query = ','.join([str(n) for n in numbers])
                params = dict(content=query, year_start=1900, year_end=2000,
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
        return (diff==0).float()*0.1/((0.01+target.float()))

