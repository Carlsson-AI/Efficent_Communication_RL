import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as lines


import torch
import torchHelpers as th


def print_cnum(t):
    return str(t['#cnum'].values[0])


def print_index(t):
    return str(t.index.values[0])

def cielab2rgb(c):
    from colormath.color_objects import LabColor, sRGBColor
    from colormath.color_conversions import convert_color

    lab = LabColor(c[0],c[1],c[2])
    rgb = convert_color(lab, sRGBColor)

    return np.array(rgb.get_value_tuple())

# basic color chip similarity

def dist(color_x, color_y):
    # CIELAB distance 76 (euclidean distance)
    diff = (color_x - color_y)
    return diff.norm(2, 1)


def sim(color_x, color_y, c = 0.001):
    # Regier similarity
    return torch.exp(-c * torch.pow(dist(color_x, color_y), 2))


# Reward functions

def basic_reward(color_codes, color_guess):
    _, I = color_guess.max(1)
    reward = (color_codes == I).float() - (color_codes != I).float()
    return reward


class WCS_Enviroment:

    def __init__(self, wcs_path='data/') -> None:
        super().__init__()

        self.color_chips = pd.read_csv(wcs_path + 'cnum-vhcm-lab-new.txt', sep='\t')
        self.cielab_map = self.color_chips[['L*', 'a*', 'b*']].values

        self.term = pd.read_csv(wcs_path + 'term.txt', sep='\t', names=['lang_num', 'spkr_num', 'chip_num', 'term_abrev'])
        self.dict = pd.read_csv(wcs_path + 'dict.txt', sep='\t', skiprows=[0], names=['lang_num', 'term_num', 'term', 'term_abrev'])
        self.term_nums = pd.merge(self.term,
                                  self.dict.drop_duplicates(subset=['lang_num', 'term_abrev']),
                                  how='inner',
                                  on=['lang_num', 'term_abrev'])

    def language_map(self, lang_num):
        l = self.term_nums.loc[self.term_nums.lang_num == lang_num]
        map = {chip_i: l.loc[l.chip_num == self.color_chips.loc[chip_i]['#cnum']]['term_num'].mode().values[0] for chip_i in range(330)}
        return map

    #Iduna (lang_num47)
    #map = language_map(47)

    def all_colors(self):
        return self.color_chips.index.values, self.cielab_map


    def batch(self, batch_size = 10):
        batch = self.color_chips.sample(n=batch_size, replace=True)

        return batch.index.values, batch[['L*', 'a*', 'b*']].values


    def color_dim(self):
        return len(self.color_chips)


    def chip_index2CIELAB(self, color_codes):
        return self.cielab_map[color_codes]


    # Printing

    def print_color_map(self, f=print_cnum, pad=3):
        # print x axsis
        print(''.ljust(pad), end="")
        for x in range(41):
            print(str(x).ljust(pad), end="")
        print('')

        # print color codes
        for y in list('ABCDEFGHIJ'):
            print(y.ljust(pad), end="")
            for x in range(41):
                t = self.color_chips.loc[(self.color_chips['H'] == x) & (self.color_chips['V'] == y)]
                if len(t) == 0:
                    s = ''
                elif len(t) == 1:
                    s = f(t)
                else:
                    raise TabError()

                print(s.ljust(pad), end="")
            print('')


    def regier_reward(self, color, color_guess, cuda):
        _, color_code_guess = color_guess.max(1)
        color_guess = th.float_var(self.chip_index2CIELAB(color_code_guess.data), cuda)
        return sim(color, color_guess)


    # plotting

    def plot_with_colors(self, V, save_to_path='dev.png', y_wcs_range='ABCDEFGHIJ', x_wcs_range=range(0, 41), use_real_color=True):
        #print_color_map(print_index, 4)

        N_x = len(x_wcs_range)
        N_y = len(y_wcs_range)
        # make an empty data set
        word = np.ones([N_y, N_x],dtype=np.int64) * -1
        rgb = np.ones([N_y, N_x, 3])
        for y_alpha, y in zip(list(y_wcs_range), range(N_y)):
            for x_wcs, x in zip(x_wcs_range, range(N_x)):
                t = self.color_chips.loc[(self.color_chips['H'] == x_wcs) & (self.color_chips['V'] == y_alpha)]
                if len(t) == 0:
                    word[y, x] = -1
                    rgb[y, x, :] = np.array([1, 1, 1])
                elif len(t) == 1:
                    word[y, x] = V[t.index.values[0]]
                    rgb[y, x, :] = cielab2rgb(t[['L*', 'a*', 'b*']].values[0])
                else:
                    raise TabError()

        fig, ax = plt.subplots(1, 1, tight_layout=True)


        my_cmap = plt.get_cmap('tab20')

        bo = 0.2
        lw = 1.5
        for y in range(N_y):
            for x in range(N_x):
                if word[y, x] >= 0:
                    word_color = my_cmap.colors[word[y, x]]
                    word_border = '-'
                    if word[y, x] != word[y, (x+1) % N_x]:
                        ax.add_line(lines.Line2D([x + 1, x + 1], [N_y - y, N_y - (y + 1)], color='w'))
                        ax.add_line(lines.Line2D([x+1-bo, x+1-bo], [N_y - (y+bo), N_y - (y+1-bo)], color=word_color, ls=word_border, lw=lw))

                    if word[y, x] != word[y, x-1 if x-1 >= 0 else N_x-1]:
                        ax.add_line(lines.Line2D([x+bo, x+bo], [N_y - (y+bo), N_y - (y+1-bo)], color=word_color, ls=word_border, lw=lw))


                    if (y+1 < N_y and word[y, x] != word[y+1, x]) or y+1 == N_y:
                        ax.add_line(lines.Line2D([x+bo, x + 1-bo], [N_y - (y + 1-bo), N_y - (y + 1-bo)], color=word_color, ls=word_border, lw=lw))
                        ax.add_line(lines.Line2D([x, x + 1], [N_y - (y + 1), N_y - (y + 1)], color='w'))

                    if (y-1 >= 0 and word[y, x] != word[y-1, x]) or y-1 < 0:
                            ax.add_line(lines.Line2D([x+bo, x + 1-bo], [N_y - (y + bo), N_y - (y + bo)], color=word_color, ls=word_border, lw=lw))


        #my_cmap = matplotlib.colors. ListedColormap(['r', 'g', 'b'])

        my_cmap.set_bad(color='w', alpha=0)
        data = rgb if use_real_color else word
        data = data.astype(np.float)
        data[data == -1] = np.nan
        ax.imshow(data, interpolation='none', cmap=my_cmap, extent=[0, N_x, 0, N_y], zorder=0)

        #ax.axis('off')
        ylabels = list(y_wcs_range)
        ylabels.reverse()
        plt.yticks([i+0.5 for i in range(len(y_wcs_range))], ylabels, fontsize=8)
        plt.xticks([i+0.5 for i in range(len(x_wcs_range))], x_wcs_range, fontsize=8)

        plt.savefig(save_to_path)
        plt.close()





