import numpy as np
from scipy import stats
import torch
import torchvision
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import copy
import evaluate
import torchHelpers as th
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# immutable(ish) game classes. Not meant to carry state between executions since each execution is based on the object
# created in the run script and not updated with new state after running on the cluster.
class BaseGame:

    def __init__(self,
                 max_epochs=1000,
                 batch_size=1000,
                 print_interval=1000,
		         evaluate_interval=0,
                 tensorboard=True,
                 log_path=''):
        super().__init__()

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.print_interval = print_interval
        self.evaluate_interval = evaluate_interval
        self.log_path = log_path
        self.training_mode = True
        self.tensorboard = tensorboard

        self.board_reward = 0
        self.sender_loss = 0
        self.receiver_loss = 0
        self.reward_log = []
        self.gibson_cost = []
        self.regier_cost = []
        self.wellformedness = []
        self.term_usage = []

    def play(self, env, agent_a, agent_b):
        agent_a = th.cuda(agent_a)
        agent_b = th.cuda(agent_b)
        optimizer = optim.Adam(list(agent_a.parameters()) + list(agent_b.parameters()), lr=0.0001)

        for i in range(self.max_epochs):
            optimizer.zero_grad()

            color_codes, colors = env.mini_batch(batch_size=self.batch_size)
            color_codes = th.long_var(color_codes)
            colors = th.float_var(colors)

            loss = self.communication_channel(env, agent_a, agent_b, color_codes, colors)
            loss.backward()
            optimizer.step()


            # Update tensorboard
            #print(self.tensorboard)
            # if((i+1) % self.print_interval == 0):
            #     self.tensorboard_update(i, env, agent_a, agent_b)
            # printing status
            if self.print_interval != 0 and ((i+1) % self.print_interval == 0):
                if self.loss_type=='REINFORCE':
                    #self.print_status(-loss)
                    self.print_status(loss)
                else:
                    self.print_status(loss)

            if self.evaluate_interval != 0 and ((i+1) % self.evaluate_interval == 0):
                self.evaluate(env, agent_a)

        #agent_a.reward_log = self.reward_log
        #agent_b.reward_log = self.reward_log

        return agent_a.cpu()

    def communication_channel(self, env, agent_a, agent_b, color_codes, colors):
        pass

    def evaluate(self, env, agent_a):
        V = evaluate.agent_language_map(env, agent_a)
        self.gibson_cost += [evaluate.compute_gibson_cost2(env, a=agent_a)[1]]
        self.regier_cost += [evaluate.communication_cost_regier(env, V=V)[0]]
        self.wellformedness += [evaluate.wellformedness(env, V=V)[0]]
        self.term_usage += [evaluate.compute_term_usage(V=V)]
        print('terms = {:2d}, gib = {:.3f}, reg = {:.3f}, well = {:.3f}'.format(self.term_usage[-1],
                                                                                self.gibson_cost[-1],
                                                                                self.regier_cost[-1],
                                                                                self.wellformedness[-1]))

        env.plot_with_colors(V, save_to_path='{}evo_map-{}_terms.png'.format(self.log_path, self.term_usage[-1]))

        plt.figure()
        plt.plot(self.gibson_cost)
        plt.savefig('{}gibson_cost_evo.png'.format(self.log_path))
        plt.figure()
        plt.plot(self.regier_cost)
        plt.savefig('{}regier_cost_evo.png'.format(self.log_path))
        plt.figure()
        plt.plot(self.wellformedness)
        plt.savefig('{}wellformedness_evo.png'.format(self.log_path))
        plt.figure()
        plt.plot(self.term_usage)
        plt.savefig('{}term_usage_evo.png'.format(self.log_path))

    def agent_language_map(self, env, a):
        pass

    # other metrics
    # Outdated as of nosy channel model
    def compute_gibson_cost(self, env, a):
        _, perceptions = env.full_batch()
        if isinstance(perceptions, np.ndarray):
            perceptions = th.float_var(torch.tensor(perceptions, dtype=torch.float32))
        perceptions = perceptions.cpu()
        all_terms = th.long_var(range(a.msg_dim), False)
        p_WC = F.softmax(a(perception=perceptions), dim=1).t().data.numpy()

        p_CW = F.softmax(a(msg=all_terms), dim=1).data.numpy()

        S = -np.diag(np.matmul(p_WC.transpose(), (np.log2(p_CW))))
        avg_S = S.sum() / len(S)  # expectation assuming uniform prior
        # debug code
        # s = 0
        # c = 43
        # for w in range(a.msg_dim):
        #     s += -p_WC[w, c]*np.log2(p_CW[w, c])
        # print(S[c] - s)
        return S, avg_S


    @staticmethod
    def reduce_maps(name, exp, reduce_method='mode'):
        maps = exp.get_flattened_results(name)
        #print(len(maps))
        if isinstance(maps, list):
            np_maps = np.array([list(map) for map in maps])
            #print(np_maps.shape)
        else:
            np_maps = np.array([list(map.values()) for map in maps])
        if reduce_method == 'mode':
            if isinstance(maps, list):
                np_mode_map = stats.mode(np_maps).mode[0]
                #print(np_mode_map)
                res = {k: np_mode_map[k] for k in range(len(maps[0]))}
                #print(res)
            else:
                np_mode_map = stats.mode(np_maps).mode[0]
                res = {k: np_mode_map[k] for k in maps[0].keys()}
        else:
            raise ValueError('unsupported reduce function: ' + reduce_method)
        return res

    @staticmethod
    def compute_ranges(V):
        lex = {}
        #for n in V.keys():
        for n in range(len(V)):
            #print(n)
            # start from 1 not 0
            if not V[n] in list(lex.keys()):
                lex[V[n]] = [n]
            else:
                lex[V[n]] += [n]
        ranges = {}
        for w in lex.keys():
            ranges[str(w)] = []
            state = 'out'
            for n in lex[w]:
                if state == 'out':
                    range_start = n
                    prev = n
                    state = 'in'
                elif state == 'in':
                    if prev + 1 != n:
                        ranges[str(w)] += [(range_start+1, prev+1)]
                        range_start = n
                    prev = n
            ranges[str(w)] += [(range_start+1, prev+1)]
        return ranges

    def print_status(self, loss):
        print("Loss %f Naive perplexity %f" %
              (loss,
               torch.exp(loss))
              )


    def tensorboard_update(self, epoch, env, a_agent, b_agent):
        # Log scalars
        writer.add_scalar('Loss/sender_loss',self.sender_loss/(self.print_interval * self.batch_size), epoch)
        writer.add_scalar('Loss/receiver_loss',self.receiver_loss/(self.print_interval * self.batch_size), epoch)
        writer.add_scalar('Metrics/Reward_'+str(self.reward_func), self.board_reward.sum()/(self.print_interval * self.batch_size), epoch)
        # log evaluation metrics
        V = evaluate.agent_language_map(env, a_agent)
        # term usage
        termed_used = evaluate.compute_term_usage(V=V)[-1]
        writer.add_scalar('Metrics/term_usage',termed_used, epoch)
        # Agent-stats
        # perception_layer = a_agent.perception_embedding.weight
        # msg_layer = a_agent.msg_creator.weight
        # writer.add_histogram('Sender/perception_layer', perception_layer, epoch)
        # writer.add_histogram('Sender/msg_layer', msg_layer, epoch)
        # writer.add_scalar('Sender/perception_layer_grad', torch.abs(perception_layer.grad).sum(), epoch)
        # writer.add_scalar('Sender/msg_layer_grad', torch.abs(msg_layer.grad).sum(), epoch)
        #
        # receiver_layer = b_agent.msg_receiver.weight
        # guess_layer = b_agent.color_estimator.weight
        # writer.add_histogram('Receiver/receiver_layer', receiver_layer, epoch)
        # writer.add_histogram('Receiver/guess_layer', guess_layer, epoch)
        # writer.add_scalar('Receiver/receiver_layer_grad', torch.abs(receiver_layer.grad).sum(), epoch)
        # writer.add_scalar('Receiver/guess_layer_grad', torch.abs(guess_layer.grad).sum(), epoch)

        # add batch
        #writer.add_text('Batch', str(self.batch), epoch)

        # Produce partition
        # if number environment:
        partition = self.compute_ranges(V)
        writer.add_text('Partition', str(partition), epoch)
        writer.flush()
        self.sender_loss=0
        self.receiver_loss=0
        # Guesses
        msg = th.float_var(np.eye(a_agent.msg_dim))
        guess_logits = b_agent(msg=msg)
        guess_probs = F.softmax(guess_logits, dim=1)
        _, guess = guess_probs.max(1)
        writer.add_text('Reciever guesses', str(guess+1), epoch)

        index, perception = env.full_batch()
        prob = F.softmax(a_agent(th.float_var(perception)),dim=1)
        prob = prob.detach().numpy()
        guess_probs = guess_probs.detach().numpy()
        for i in range(perception.shape[0]):
            fig,ax = plt.subplots(figsize=(5,5))
            plt.plot(range(a_agent.msg_dim) ,prob[i, :])
            writer.add_figure('prob_words' + str(i + 1) + '/sender' ,fig, epoch)

        for i in range(guess_probs.shape[0]):
            fig,ax = plt.subplots(figsize=(5,5))
            plt.plot(range(guess_probs.shape[1]) ,guess_probs[i, :])
            writer.add_figure('prob_guess' + str(i + 1) + '/receiver', fig, epoch)



        #writer.add_histogram('Sender_prob/' + str(i+1), prob, epoch)



class NoisyChannelGame(BaseGame):

    def __init__(self,
                 reward_func='regier_reward',
                 bw_boost=0,
                 com_noise=0,
                 msg_dim=11,
                 max_epochs=1000,
                 perception_noise=0,
                 batch_size=100,
                 print_interval=1000,
                 evaluate_interval=0,
                 log_path='',
                 perception_dim=3,
                 entropy_coef=0,
                 loss_type='CrossEntropyLoss'):
        super().__init__(max_epochs, batch_size, print_interval, evaluate_interval, log_path)
        self.reward_func = reward_func
        self.bw_boost = bw_boost
        self.com_noise = com_noise
        self.msg_dim = msg_dim
        self.perception_noise = perception_noise
        self.perception_dim = perception_dim
        self.reward_log = []
        self.loss_type = loss_type

        self.sum_reward = 0
        self.n_points = 0
        self.baseline = 0
        self.entropy_coef = entropy_coef

    def communication_channel(self, env, agent_a, agent_b, target, perception):
        # add perceptual noise
        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.perception_dim),
                                    torch.ones(self.batch_size, self.perception_dim) * self.perception_noise).sample())
        perception = perception + noise
        # generate message
        msg_logits = agent_a(perception=perception)
        # msg_probs = F.gumbel_softmax(msg_logits, tau=2 / 3)
        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.msg_dim),
                                    torch.ones(self.batch_size, self.msg_dim) * self.com_noise).sample())
        msg_probs = F.softmax(msg_logits + noise, dim=1)
        msg_dist =  Categorical(msg_probs)
        msg = msg_dist.sample()
        # interpret message and sample a guess
        # discrete training
        guess_logits = agent_b(msg=msg)
        guess_probs = F.softmax(guess_logits, dim=1)
        #guess_probs = F.gumbel_softmax(msg_logits, tau=10 / 3, dim=1)
        m = Categorical(guess_probs)
        guess = m.sample()



        #compute reward
        if self.reward_func == 'regier_reward':
            CIELAB_guess = env.chip_index2CIELAB(guess.data)
            reward = env.regier_reward(perception, CIELAB_guess, bw_boost=self.bw_boost)
        elif self.reward_func == 'abs_dist':
            diff = torch.abs(target - (1 + guess.unsqueeze(dim=1)))
            reward = 1-(diff.float()/100)  #1-(diff.float()/50)
        elif self.reward_func == 'abs_penalty':
            diff = torch.abs(target - (1 + guess.unsqueeze(dim=1)))
            reward = 1-(diff.float()/100)
            # Check whether the reciver assigns more than one word to each number
            reward = reward + env.word2number(agent_b)
        elif self.reward_func == 'exp_reward':
            diff = torch.abs(target - guess.unsqueeze(dim=1))
            reward = 2 ** (-0.1 * diff.float())
        elif self.reward_func == 'sim_index':
            reward = env.sim_index(target, guess)
        self.sum_reward += reward.sum()
        self.board_reward = reward
        # compute loss and update model
        if self.loss_type =='REINFORCE':

            # compute baseline
            self.n_points += 1
            self.baseline += (reward.mean() - self.baseline) / self.n_points
            # receiver_loss =  self.criterion_receiver(guess_logits, target.squeeze())
            sender_loss = (-msg_dist.log_prob(msg) * (reward - self.baseline)).sum() / self.batch_size
            receiver_loss = (-m.log_prob(guess) * (reward - self.baseline)).sum() / self.batch_size
            # For tensorboard logging
            self.sender_loss += sender_loss
            self.receiver_loss += receiver_loss
            loss = receiver_loss + sender_loss
        elif self.loss_type == 'CrossEntropyLoss':
            loss = self.criterion_receiver(guess_logits, target.squeeze())
            # For tensorboard logging
            self.receiver_loss += loss
        return loss

    def print_status(self, loss):

        print("Loss %f Average reward %f" %
              (loss, self.sum_reward / (self.print_interval * self.batch_size))
              )
        #print("Reward_log: " + str(self.reward_log))
        self.sum_reward = 0

class OneHotChannelGame(BaseGame):
    def __init__(self,
                 reward_func='regier_reward',
                 sender_loss_multiplier=100,
                 msg_dim=11,
                 max_epochs=1000,
                 perception_noise=0,
                 batch_size=100,
                 print_interval=1000,
                 perception_dim=3):
        super().__init__(max_epochs, batch_size, print_interval)
        self.reward_func = reward_func
        self.sender_loss_multiplier = sender_loss_multiplier
        self.msg_dim = msg_dim
        self.perception_noise = perception_noise
        self.perception_dim = perception_dim

        self.criterion_receiver = torch.nn.CrossEntropyLoss()
        self.sum_reward = 0

    def communication_channel(self, env, agent_a, agent_b, target, perception):
        # add perceptual noise
        if self.training_mode:
            noise = th.float_var(Normal(torch.zeros(self.batch_size, self.perception_dim),
                                        torch.ones(self.batch_size, self.perception_dim) * self.perception_noise).sample())
            perception = perception + noise
        # Sample message
        probs = agent_a(perception=perception)
        m = Categorical(probs)
        msg = m.sample()
        # interpret message
        guess = agent_b(msg=msg)
        # compute reward
        if self.reward_func == 'basic_reward':
            reward = env.basic_reward(target, guess)
        elif self.reward_func == 'regier_reward':
            reward = env.regier_reward(perception, guess)
        elif self.reward_func == 'number_reward':
            reward = env.number_reward(target, guess)
        elif self.reward_func == 'inverted_reward':
            reward = env.inverted_number(target, guess)
        self.sum_reward += reward.sum()
        # compute loss
        self.loss_sender = self.sender_loss_multiplier * ((-m.log_prob(msg) * reward).sum() / self.batch_size)
        self.loss_receiver = self.criterion_receiver(guess, target)
        return self.loss_receiver + self.loss_sender

    def print_status(self, loss):

        print("Loss sender %f Loss receiver %f Naive perplexity %f Average reward %f" %
              (self.loss_sender,
               self.loss_receiver,
               torch.exp(self.loss_receiver), self.sum_reward / (self.print_interval * self.batch_size))
              )
        self.sum_reward = 0

class MultiTaskGame(NoisyChannelGame):
    def __init__(self,
                 reward_func='regier_reward',
                 bw_boost=0,
                 com_noise=0,
                 msg_dim=11,
                 max_epochs=1000,
                 perception_noise=0,
                 batch_size=100,
                 print_interval=1000,
                 evaluate_interval=0,
                 log_path='',
                 perception_dim=3,
                 loss_type='CrossEntropyLoss'):
        super().__init__(reward_func, bw_boost, com_noise, msg_dim, max_epochs, perception_dim, batch_size, print_interval, evaluate_interval, log_path, perception_dim, loss_type)


        #self.criterion_receiver = torch.nn.CrossEntropyLoss()
    def play(self, env, agent_a, agent_b):
        agent_a = th.cuda(agent_a)
        agent_b = th.cuda(agent_b)

        optimizer = optim.Adam(list(agent_a.parameters()) + list(agent_b.parameters()), lr=0.0001)

        for i in range(self.max_epochs):
            optimizer.zero_grad()
            # Agent a sends a message
            color_codes, colors = env.mini_batch(batch_size=self.batch_size)
            color_codes = th.long_var(color_codes)
            colors = th.float_var(colors)
            loss1 = self.communication_channel(env, agent_a, agent_b, color_codes, colors)
            loss1.backward()
            # Agent b sends a message
            color_codes, colors = env.mini_batch(batch_size=self.batch_size)
            color_codes = th.long_var(color_codes)
            colors = th.float_var(colors)
            loss2 = self.communication_channel(env, agent_b, agent_a, color_codes, colors)
            loss2.backward()
            # Backprogate
            #loss.backward()
            optimizer.step()
            loss = loss1 + loss2
            # printing status
            if self.print_interval != 0 and ((i+1) % self.print_interval == 0):
                #self.tensorboard_update(i, env, agent_a)
                self.print_status(loss)

            if self.evaluate_interval != 0 and ((i+1) % self.evaluate_interval == 0):
                self.evaluate(env, agent_a)


        return agent_a.cpu()

class UpdatedNoisyChannelGame(NoisyChannelGame):

    def __init__(self,
                 reward_func='regier_reward',
                 bw_boost=0,
                 com_noise=0,
                 msg_dim=11,
                 max_epochs=1000,
                 perception_noise=0,
                 batch_size=100,
                 print_interval=1000,
                 evaluate_interval=0,
                 log_path='',
                 perception_dim=3,
                 entropy_coef=0.1,
                 loss_type='REINFORCE'):
        super().__init__(reward_func, bw_boost, com_noise, msg_dim, max_epochs, perception_dim, batch_size, print_interval, evaluate_interval, log_path, perception_dim, loss_type)
        self.loss_type = 'REINFORCE'
        self.entropy_coef = 0

    def play(self, env, agent_a, agent_b):
        agent_a = th.cuda(agent_a)
        agent_b = th.cuda(agent_b)
        receiver_opt = optim.Adam(list(agent_b.parameters()))
        optimizer = optim.Adam(list(agent_a.parameters()) + list(agent_b.parameters()))

        for i in range(self.max_epochs):
            for j in range(50):
                color_codes, colors = env.mini_batch(batch_size=self.batch_size)
                color_codes = th.long_var(color_codes)
                colors = th.float_var(colors)
                receiver_loss, _, _ = self.communication_channel(env, agent_a, agent_b, color_codes, colors)
                receiver_loss.backward()
                receiver_opt.step()
                receiver_opt.zero_grad()
            self.board_reward = 0
            optimizer.zero_grad()

            color_codes, colors = env.mini_batch(batch_size=self.batch_size)
            color_codes = th.long_var(color_codes)
            colors = th.float_var(colors)

            receiver_loss, sender_loss, entropy_loss = self.communication_channel(env, agent_a, agent_b, color_codes, colors)
            loss = receiver_loss + sender_loss + entropy_loss
            loss.backward()
            optimizer.step()



            # Update tensorboard
            #print(self.tensorboard)
            if((i+1) % self.print_interval == 0):
                self.tensorboard_update(i, env, agent_a, agent_b)
            # printing status
            if self.print_interval != 0 and ((i+1) % self.print_interval == 0):
                if self.loss_type=='REINFORCE':
                    #self.print_status(-loss)
                    self.print_status(loss)
        #        else:
                    #self.print_status(loss)

            if self.evaluate_interval != 0 and ((i+1) % self.evaluate_interval == 0):
                self.evaluate(env, agent_a)
    def communication_channel(self, env, agent_a, agent_b, target, perception):
        # add perceptual noise
        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.perception_dim),
                                    torch.ones(self.batch_size, self.perception_dim) * self.perception_noise).sample())
        perception = perception + noise
        # generate message
        msg_logits = agent_a(perception=perception)
        # msg_probs = F.gumbel_softmax(msg_logits, tau=2 / 3)
        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.msg_dim),
                                    torch.ones(self.batch_size, self.msg_dim) * self.com_noise).sample())
        msg_probs = F.softmax(msg_logits + noise, dim=1)
        #ßßßßmsg_probs = F.gumbel_softmax(msg_logits + noise, tau=10 / 3, dim=1)
        msg_dist =  Categorical(msg_probs)
        msg = msg_dist.sample()
        # interpret message and sample a guess
        guess_logits = agent_b(msg=msg_probs)
        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.msg_dim),
                                    torch.ones(self.batch_size, self.msg_dim) * self.com_noise).sample())
        guess_probs = F.softmax(guess_logits, dim=1)
        #guess_probs = F.gumbel_softmax(msg_logits, tau=10 / 3, dim=1)
        m = Categorical(guess_probs)
        guess = m.sample()

        #compute reward
        if self.reward_func == 'regier_reward':
            CIELAB_guess = env.chip_index2CIELAB(guess.data)
            reward = env.regier_reward(perception, CIELAB_guess, bw_boost=self.bw_boost)
        elif self.reward_func == 'abs_dist':
            diff = torch.abs(target - (1 + guess.unsqueeze(dim=1)))
            reward = 1-(diff.float()/100)  #1-(diff.float()/50)
        elif self.reward_func == 'abs_penalty':
            diff = torch.abs(target - (1 + guess.unsqueeze(dim=1)))
            reward = 1-(diff.float()/100)
            # Check whether the reciver assigns more than one word to each number
            reward = reward + env.word2number(agent_b)
        elif self.reward_func == 'exp_reward':
            diff = torch.abs(target - guess.unsqueeze(dim=1))
            reward = 2 ** (-0.1 * diff.float())
        elif self.reward_func == 'sim_index':
            reward = env.sim_index(target, guess)
        self.sum_reward += reward.sum()
        self.board_reward = reward
        # compute loss and update model
        if self.loss_type =='REINFORCE':
            # compute baseline
            self.n_points += 1
            self.baseline += (reward.mean() - self.baseline) / self.n_points
            # receiver_loss =  self.criterion_receiver(guess_logits, target.squeeze())
            sender_loss = (-msg_dist.log_prob(msg) * (reward - self.baseline)).sum() / self.batch_size
            receiver_loss = (-m.log_prob(guess) * (reward - self.baseline)).sum() / self.batch_size
            entropy_loss =  -(self.entropy_coef * (1 * msg_dist.entropy().mean() + 3 * m.entropy().mean()))
            # For tensorboard logging
            self.sender_loss += sender_loss
            self.receiver_loss += receiver_loss
            #self.entropy_coef = 0.999 * self.entropy_coef
            # loss = receiver_loss + sender_loss
            #loss = receiver_loss
            return receiver_loss, sender_loss, entropy_loss
        elif self.loss_type == 'CrossEntropyLoss':
            loss = self.criterion_receiver(guess_logits, target.squeeze())
            # For tensorboard logging
            self.receiver_loss += loss
        return receiver_loss, sender_loss, entropy_loss

class GumbelSoftmaxGame(BaseGame):

    def __init__(self,
                 reward_func='abs_dist',
                 bw_boost=0,
                 com_noise=0,
                 msg_dim=11,
                 max_epochs=1000,
                 perception_noise=0,
                 batch_size=100,
                 print_interval=1000,
                 evaluate_interval=0,
                 log_path='',
                 perception_dim=3,
                 loss_type='CrossEntropyLoss'):
        super().__init__(max_epochs, batch_size, print_interval, evaluate_interval, log_path)
        self.reward_func = reward_func
        self.bw_boost = bw_boost
        self.com_noise = com_noise
        self.msg_dim = msg_dim
        self.perception_noise = perception_noise
        self.perception_dim = perception_dim
        self.reward_log = []
        self.loss_type = loss_type

        self.sum_reward = 0

        self.criterion_receiver = torch.nn.CrossEntropyLoss()

    def communication_channel(self, env, agent_a, agent_b, target, perception):
        # add perceptual noise

        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.perception_dim),
                                    torch.ones(self.batch_size, self.perception_dim) * self.perception_noise).sample())

        perception = perception + noise
        # generate message
        msg_logits = agent_a(perception=perception)
        # msg_probs = F.gumbel_softmax(msg_logits, tau=2 / 3)
        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.msg_dim),
                                    torch.ones(self.batch_size, self.msg_dim) * self.com_noise).sample())

        msg_probs = F.softmax(msg_logits + noise, dim=1)
        #msg_probs = F.gumbel_softmax(msg_logits + noise, tau=10 / 3, dim=1)
        msg_dist =  Categorical(msg_probs)
        msg = msg_dist.sample()
        # interpret message and sample a guess
        guess_logits = agent_b(msg=msg)
        guess_probs = F.softmax(guess_logits, dim=1)
        #guess_probs = F.gumbel_softmax(msg_logits, tau=10 / 3, dim=1)
        m = Categorical(guess_probs)
        guess = m.sample()

        #compute reward
        if self.reward_func == 'regier_reward':
            CIELAB_guess = env.chip_index2CIELAB(guess.data)
            reward = env.regier_reward(perception, CIELAB_guess, bw_boost=self.bw_boost)
        elif self.reward_func == 'abs_dist':
            diff = torch.abs(target.unsqueeze(dim=1) - guess.unsqueeze(dim=1))
            reward = 1-(diff.float()/100)  #1-(diff.float()/50)

        # compute loss and update model
        if self.loss_type =='REINFORCE':
            sender_loss = -(reward * msg_dist.log_prob(msg)).sum() / self.batch_size
            receiver_loss = -(reward * m.log_prob(guess)).sum() / self.batch_size
            #receiver_loss = self.criterion_receiver(guess_logits, target.squeeze())

            loss = receiver_loss + sender_loss

        elif self.loss_type == 'CrossEntropyLoss':
            loss = self.criterion_receiver(guess_logits, target.squeeze())
        return loss

    def print_status(self, loss):

        print("Loss %f Average reward %f" %
              (loss, self.sum_reward / (self.print_interval * self.batch_size))
              )
        #print("Reward_log: " + str(self.reward_log))
        self.sum_reward = 0

class ReconstructChannelGame(NoisyChannelGame):
    def __init__(self,
                 reward_func='regier_reward',
                 bw_boost=0,
                 com_noise=0,
                 msg_dim=11,
                 max_epochs=1000,
                 perception_noise=0,
                 batch_size=100,
                 print_interval=1000,
                 evaluate_interval=0,
                 log_path='',
                 perception_dim=3,
                 recon_param=0.1,
                 loss_type='CrossEntropyLoss'):
        super().__init__(reward_func, bw_boost, com_noise, msg_dim, max_epochs, perception_dim, batch_size, print_interval, evaluate_interval, log_path, perception_dim, loss_type)
        self.recon_param = recon_param
    def communication_channel(self, env, agent_a, agent_b, target, perception):
        # add perceptual noise

        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.perception_dim),
                                    torch.ones(self.batch_size, self.perception_dim) * self.perception_noise).sample())
        perception = perception + noise
        # generate message
        msg_logits = agent_a(perception=perception)
        # msg_probs = F.gumbel_softmax(msg_logits, tau=2 / 3)
        noise = th.float_var(Normal(torch.zeros(self.batch_size, self.msg_dim),
                                    torch.ones(self.batch_size, self.msg_dim) * self.com_noise).sample())
        msg_probs = F.softmax(msg_logits + noise, dim=1)
        #ßßßßmsg_probs = F.gumbel_softmax(msg_logits + noise, tau=10 / 3, dim=1)
        msg_dist =  Categorical(msg_probs)
        msg = msg_dist.sample()
        # interpret message and sample a guess
        guess_logits = agent_b(msg=msg)
        guess_probs = F.softmax(guess_logits, dim=1)
        #guess_probs = F.gumbel_softmax(msg_logits, tau=10 / 3, dim=1)
        m = Categorical(guess_probs)
        guess = m.sample()

        # Reconstruct( Sanity check)
        recon_logits = agent_a(msg=msg)
        recon_probs = F.softmax(recon_logits, dim=1)
        recon_dist = Categorical(recon_probs)
        recon_guess = recon_dist.sample()
        # CrossEntropy or REINFORCE?
        # This becomes a standard autoencoder ?
        recon_diff = torch.abs(target - recon_guess.unsqueeze(dim=1))
        recon_reward = 1-(recon_diff.float()/100)
        recon_loss = 0.5*(-recon_dist.log_prob(recon_guess) * recon_reward).sum() / self.batch_size
        #recon_loss =self.recon_param * self.criterion_receiver(recon_logits, target.squeeze())
        #compute reward
        if self.reward_func == 'regier_reward':
            CIELAB_guess = env.chip_index2CIELAB(guess.data)
            reward = env.regier_reward(perception, CIELAB_guess, bw_boost=self.bw_boost)
        elif self.reward_func == 'abs_dist':
            diff = torch.abs(target - guess.unsqueeze(dim=1))
            reward = 1-(diff.float()/100)  #1-(diff.float()/50)
            #reward = 1 /(diff.float()+1)**2
        elif self.reward_func == 'exp_reward':
            diff = torch.abs(target - guess.unsqueeze(dim=1))
            reward = 2 ** (-diff.float())  #1-(diff.float()/50)
            #reward = 1 /(diff.float()+1)**2
        elif self.reward_func == 'number_reward':
            reward = env.number_reward(target, guess)
        elif self.reward_func == 'inverted_reward':
            reward = env.inverted_number(target, guess)
        elif self.reward_func == 'interval_reward':
            reward = env.interval_reward(target, guess)
        elif self.reward_func == 'target_reward':
            reward = env.target_reward(target, guess)
        elif self.reward_func == 'sim_index':
            reward = env.sim_index(target, guess)
        self.sum_reward += reward.sum()
        self.board_reward = reward
        # compute loss and update model
        if self.loss_type =='REINFORCE':
            #receiver_loss =  self.criterion_receiver(guess_logits, target.squeeze())
            sender_loss = (-msg_dist.log_prob(msg) * reward).sum() / self.batch_size
            receiver_loss = (-m.log_prob(guess) * reward).sum() / self.batch_size
            # For tensorboard logging
            #entropy_loss =  -(self.entropy_coef * (msg_dist.entropy().mean() + m.entropy().mean()))
            self.sender_loss += sender_loss
            self.receiver_loss += receiver_loss
            #self.entropy_coef = 0.999 * self.entropy_coef
            loss = receiver_loss + sender_loss + recon_loss
            # loss = receiver_loss + sender_loss + entropy_loss
            #loss = receiver_loss
        elif self.loss_type == 'CrossEntropyLoss':
            loss = self.criterion_receiver(guess_logits, target.squeeze())
            # For tensorboard logging
            self.receiver_loss += loss
        return loss

    def print_status(self, loss):

        print("Loss %f Average reward %f" %
              (loss, self.sum_reward / (self.print_interval * self.batch_size))
              )
        #print("Reward_log: " + str(self.reward_log))
        self.sum_reward = 0
