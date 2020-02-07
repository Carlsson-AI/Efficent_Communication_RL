import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchHelpers as th
import numpy as np
class BasicAgent(nn.Module):

    def __init__(self, msg_dim, hidden_dim, color_dim, perception_dim):
        super().__init__()
        self.msg_dim = msg_dim

        # Receiving part
        self.msg_receiver = nn.Embedding(msg_dim, hidden_dim)
        self.color_estimator = nn.Linear(hidden_dim, color_dim)

        #Sending part
        self.perception_embedding = nn.Linear(perception_dim, hidden_dim)
        self.msg_creator = nn.Linear(hidden_dim, msg_dim)

    def forward(self, perception=None, msg=None, tau=1):

        if msg is not None:
            h = F.tanh(self.msg_receiver(msg))
            color_logits = self.color_estimator(h)
            return color_logits

        if perception is not None:
            h = F.tanh(self.perception_embedding(perception))

            probs = F.softmax(self.msg_creator(h)/tau, dim=1)
            return probs


class SoftmaxAgent(nn.Module):
    def __init__(self, msg_dim, hidden_dim, color_dim, perception_dim):
        super().__init__()
        self.msg_dim = msg_dim
        self.perception_dim = perception_dim
        self.color_dim = color_dim
        # Quick fix for logging reward, TO BE FIXED
        self.reward_log = []
        #Sending part
        self.perception_embedding = nn.Linear(perception_dim, hidden_dim)
        self.s1 = nn.Linear(hidden_dim, hidden_dim)
        self.s2 = nn.Linear(hidden_dim, hidden_dim)
        self.s3 = nn.Linear(hidden_dim, hidden_dim)
        self.s4 = nn.Linear(hidden_dim, hidden_dim)
        self.s5 = nn.Linear(hidden_dim, hidden_dim)
        self.s6 = nn.Linear(hidden_dim, hidden_dim)
        self.msg_creator = nn.Linear(hidden_dim, msg_dim)

        # Receiving part
        # self.msg_receiver = nn.Linear(msg_dim, hidden_dim)
        self.msg_receiver = nn.Linear(msg_dim, hidden_dim)
        self.r1 = nn.Linear(hidden_dim, hidden_dim)
        self.r2 = nn.Linear(hidden_dim, hidden_dim)
        self.r3 = nn.Linear(hidden_dim, hidden_dim)
        self.r4 = nn.Linear(hidden_dim, hidden_dim)
        self.r5 = nn.Linear(hidden_dim, hidden_dim)
        self.r6 = nn.Linear(hidden_dim, hidden_dim)
        self.color_estimator = nn.Linear(hidden_dim, color_dim)

    def forward(self, perception=None, msg=None, tau=1/3, test_time=False):

        if perception is not None:
            h = F.relu(self.perception_embedding(perception))
            h = F.relu(self.s1(h))
            logits = self.msg_creator(h)

            return logits

        if msg is not None:
            # First make discrete input into a onehot distribution (used for eval)
            if msg.data.type() == 'torch.LongTensor':
                onehot = torch.FloatTensor(len(msg), self.msg_dim)
                onehot.zero_()
                msg = Variable(onehot.scatter_(1, msg.data.unsqueeze(1), 1))

            h = F.relu(self.msg_receiver(msg))
            color_logits = self.color_estimator(h)
            return color_logits

class BasicMultiTaskAgent(nn.Module):
    def __init__(self, msg_dim, hidden_dim, shared_dim, color_dim, perception_dim):
        super().__init__()
        self.msg_dim = msg_dim


        # Shared part
        self.shared_layer1 = nn.Linear(hidden_dim, shared_dim)
        self.shared_layer2 = nn.Linear(shared_dim, shared_dim)
        # Receiving part
        self.msg_receiver = nn.Linear(msg_dim, hidden_dim)
        self.color_estimator = nn.Linear(shared_dim, color_dim)

        #Sending part
        self.perception_embedding = nn.Linear(perception_dim, hidden_dim)
        self.msg_creator = nn.Linear(shared_dim, msg_dim)

    def forward(self, perception=None, msg=None, tau=1):

        if msg is not None:
            if msg.data.type() == 'torch.LongTensor':
                onehot = torch.FloatTensor(len(msg), self.msg_dim)
                onehot.zero_()
                msg = Variable(onehot.scatter_(1, msg.data.unsqueeze(1), 1))
            h = F.relu(self.msg_receiver(msg))
            h = F.relu(self.shared_layer1(h))
            h = F.relu(self.shared_layer2(h))
            color_logits = self.color_estimator(h)
            return color_logits

        if perception is not None:
            h = F.relu(self.perception_embedding(perception))
            h = F.relu(self.shared_layer1(h))
            h = F.relu(self.shared_layer2(h))
            probs = F.relu(self.msg_creator(h))
            return probs

class SimpleAgent(nn.Module):
    def __init__(self, msg_dim, hidden_dim, color_dim, perception_dim):
        super().__init__()
        self.msg_dim = msg_dim
        self.perception_dim = perception_dim
        self.color_dim = color_dim
        # Quick fix for logging reward, TO BE FIXED
        self.reward_log = []
        #Sending part
        self.msg_creator = nn.Linear(perception_dim, msg_dim)

        # Receiving part
        # self.msg_receiver = nn.Linear(msg_dim, hidden_dim)
        self.color_estimator = nn.Linear(msg_dim, color_dim)
        # self.color_estimator = nn.Embedding(msg_dim, color_dim)

    def forward(self, perception=None, msg=None, tau=1/3, test_time=False):

        if perception is not None:
            logits = self.msg_creator(perception)

            return logits

        if msg is not None:
            # First make discrete input into a onehot distribution (used for eval)
            if msg.data.type() == 'torch.LongTensor':
                onehot = torch.FloatTensor(len(msg), self.msg_dim)
                onehot.zero_()
                msg = Variable(onehot.scatter_(1, msg.data.unsqueeze(1), 1))
            color_logits = self.color_estimator(msg)
            return color_logits
    def __init__(self, msg_dim, hidden_dim, color_dim, perception_dim):
        super().__init__()
        self.msg_dim = msg_dim
        self.perception_dim = perception_dim
        self.color_dim = color_dim
        # Quick fix for logging reward, TO BE FIXED
        self.reward_log = []
        #Sending part
        self.msg_creator = nn.Linear(perception_dim, msg_dim)

        # Receiving part
        self.color_estimator = nn.Linear(msg_dim, color_dim)

    def forward(self, perception=None, msg=None, tau=1/3, test_time=False):

        if perception is not None:
            logits = self.msg_creator(perception)

            return logits

        if msg is not None:
            # First make discrete input into a onehot distribution (used for eval)
            if msg.data.type() == 'torch.LongTensor':
                onehot = torch.FloatTensor(len(msg), self.msg_dim)
                onehot.zero_()
                msg = Variable(onehot.scatter_(1, msg.data.unsqueeze(1), 1))
            color_logits = self.color_estimator(msg)
            return color_logits
class DummyAgent(nn.Module):
    def __init__(self, msg_dim, hidden_dim, color_dim, perception_dim):
        super().__init__()
        self.msg_dim = msg_dim
        self.perception_dim = perception_dim
        self.color_dim = color_dim
        # Quick fix for logging reward, TO BE FIXED
        self.reward_log = []
        #Sending part
        self.perception_embedding = nn.Linear(perception_dim, hidden_dim)
        self.msg_creator = nn.Linear(hidden_dim, msg_dim)

        # Receiving part
        # self.msg_receiver = nn.Linear(msg_dim, hidden_dim)
        self.msg_receiver = nn.Linear(msg_dim, hidden_dim)
        self.color_estimator = nn.Linear(hidden_dim, color_dim)

    def forward(self, perception=None, msg=None, tau=1/3, test_time=False):

        if perception is not None:
            numbers = perception.detach().numpy()
            batch_size = numbers.shape[0]
            array = np.zeros([batch_size,self.msg_dim])
            if batch_size == 1:
                array[0, int(numbers[0][0]) -1 ] = 10
            else:
                for i in range(batch_size):
                    array[i, int(numbers[i][0]) -1 ] = 10
            logits = torch.FloatTensor(array)
            return logits

        if msg is not None:
            # First make discrete input into a onehot distribution (used for eval)
            if msg.data.type() == 'torch.LongTensor':
                onehot = torch.FloatTensor(len(msg), self.msg_dim)
                onehot.zero_()
                msg = Variable(onehot.scatter_(1, msg.data.unsqueeze(1), 1))

            h = F.relu(self.msg_receiver(msg))
            color_logits = self.color_estimator(h)
            return color_logits
