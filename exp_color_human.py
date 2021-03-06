import numpy as np

import Correlation_Clustering
import gridengine as sge
import com_game
import viz
import evaluate
from gridengine.pipeline import Experiment
import com_enviroments
import agents
import exp_shared

import matplotlib.pyplot as plt

def run(host_name='local', pipeline=''):
    if pipeline != '':
        return exp_shared.load_exp(pipeline)

    wcs = com_enviroments.make('wcs')

    # Create and run new experiment
    exp = Experiment(exp_name='human',
                     fixed_params=[('env', 'wcs')],
                     param_ranges=[('lang_id', list(wcs.human_mode_maps.keys()))]) # range(1, 5))]) #range(1, 5))]) #



    exp_i = 0
    for (params_i, params_v) in exp:
        print('Scheduled %d experiments out of %d' % (exp_i, len(list(exp))))
        exp_i += 1

        map = wcs.human_mode_maps[params_v[exp.axes['lang_id']]]

        exp.set_result('language_map', params_i, map)
        exp.set_result('term_usage', params_i, exp.run(evaluate.compute_term_usage, V=map).result())
        exp.set_result('regier_cost', params_i, exp.run(evaluate.regier2, wcs, map=map).result())
        #exp.set_result('regier_cost', params_i, exp.run(evaluate.communication_cost_regier, wcs, V=map, sum_over_whole_s=True).result())
        exp.set_result('wellformedness', params_i, exp.run(evaluate.wellformedness, wcs, V=map).result())


    exp.save()

    return exp


def visualize(exp):
    regier_cost = exp.reshape('regier_cost', as_function_of_axes=['lang_id'])
    term_usage = exp.reshape('term_usage', as_function_of_axes=['lang_id'])

    plt.figure()
    plt.scatter(term_usage, regier_cost)
    plt.show()
    fig_name = exp.pipeline_path + '/fig_regier_wcs_scatter.png'
    plt.savefig(fig_name)


    #viz.plot_with_conf2(exp, 'regier_cost', 'term_usage', 'lang_id')
    #viz.plot_with_conf2(exp, 'wellformedness', 'term_usage', 'lang_id')

    maps = exp.reshape('language_map')
    term_usage = exp.reshape('term_usage')
    iterations = 10
    e = com_enviroments.make('wcs')
    for t in np.unique(term_usage):
        if len(maps[term_usage == t]) >= 1:
            consensus_map = Correlation_Clustering.compute_consensus_map(maps[term_usage == t], k=t, iter=iterations)
            e.plot_with_colors(consensus_map,
                               save_to_path=exp.pipeline_path + 'human_consensus_map-' + str(t) + '_terms.png')

def main(args):
    # Run experiment
    exp = run(args.host_name, args.pipeline)

    visualize(exp)

if __name__ == "__main__":
    main(exp_shared.parse_script_arguments().parse_args())

