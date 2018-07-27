import numpy as np
import gridengine as sge
import com_game
import viz
from gridengine.pipeline import Experiment


import com_enviroments
import agents
import exp_shared


def run(host_name):
    # Create and run new experiment
    queue = exp_shared.create_queue(host_name)
    queue.sync('.', '.', exclude=['pipelines/*', 'fig/*', 'old/*', 'cogsci/*'], sync_to=sge.SyncTo.REMOTE,
               recursive=True)
    exp = Experiment(exp_name='color_s',
                     fixed_params=[('env', 'wcs'),
                                   ('max_epochs', 1000),  # 10000
                                   ('hidden_dim', 20),
                                   ('batch_size', 100),
                                   ('perception_dim', 3),
                                   ('target_dim', 330),
                                   ('print_interval', 1000)],
                     param_ranges=[('avg_over', range(1)),  # 50
                                   ('perception_noise', [0, 50]),  # [0, 25, 50, 100],
                                   ('msg_dim', range(3, 5)),  # 3, 12
                                   ('com_noise', np.linspace(start=0, stop=0.5, num=1))],  # 10
                     queue=queue)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.REMOTE, recursive=True)

    env = exp.run(com_enviroments.make, exp.fixed_params['env']).result()
    exp_i = 0
    for (params_i, params_v) in exp:
        print('Scheduled %d experiments out of %d' % (exp_i, len(list(exp))))
        exp_i += 1

        agent_a = agent_b = agents.SoftmaxAgent(msg_dim=params_v[exp.axes['msg_dim']],
                                                hidden_dim=exp.fixed_params['hidden_dim'],
                                                color_dim=exp.fixed_params['target_dim'],
                                                perception_dim=exp.fixed_params['perception_dim'])

        game = com_game.NoisyChannelGame(com_noise=params_v[exp.axes['com_noise']],
                                         msg_dim=params_v[exp.axes['msg_dim']],
                                         max_epochs=exp.fixed_params['max_epochs'],
                                         perception_noise=params_v[exp.axes['perception_noise']],
                                         batch_size=exp.fixed_params['batch_size'],
                                         print_interval=exp.fixed_params['print_interval'])

        game_outcome = exp.run(game.play, env, agent_a, agent_b).result()

        V = exp.run(game.agent_language_map, env, a=game_outcome).result()

        exp.set_result('agent_language_map', params_i, V)
        exp.set_result('gibson_cost', params_i, exp.run(game.compute_gibson_cost, env, a=game_outcome).result(1))
        exp.set_result('regier_cost', params_i, exp.run(game.communication_cost_regier, env, V=V).result())
        exp.set_result('wellformedness', params_i, exp.run(game.wellformedness, env, V=V).result())
        exp.set_result('term_usage', params_i, exp.run(game.compute_term_usage, V=V).result())
    exp.save()
    print("\nAll tasks queued to clusters")

    # wait for all tasks to complete
    exp.wait(retry_interval=5)
    queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)

    return exp


def visualize(exp):
    print('plot results')


    viz.plot_with_conf(exp,
                    'gibson_cost', 'com_noise', 'msg_dim',
                    measure_label='Gibson communication efficiency',
                    x_label='Communication noise',
                    z_label='terms')
    viz.plot_with_conf(exp, 'gibson_cost', 'perception_noise', 'msg_dim')
    viz.plot_with_conf(exp, 'gibson_cost', 'com_noise', 'perception_noise')

    viz.plot_with_conf(exp, 'regier_cost', 'com_noise', 'msg_dim')
    viz.plot_with_conf(exp, 'wellformedness', 'com_noise', 'msg_dim')
    viz.plot_with_conf(exp, 'term_usage', 'com_noise', 'msg_dim')
    viz.plot_with_conf(exp, 'term_usage', 'perception_noise', 'msg_dim')
    viz.plot_with_conf(exp, 'term_usage', 'perception_noise', 'com_noise')

    #exp.run(env, call_member='plot_with_colors', V=V, save_to_path=exp.pipeline_path + 'language_map.png')


def main():
    args = exp_shared.parse_script_arguments()
    # Run experiment
    if args.pipeline == '':
        exp = run(args.host_name)
    else:
        # Load existing experiment
        exp = Experiment.load(args.pipeline)
        if args.resync == 'y':
            exp.wait(retry_interval=5)
            exp.queue.sync(exp.pipeline_path, exp.pipeline_path, sync_to=sge.SyncTo.LOCAL, recursive=True)
    # Visualize experiment
    visualize(exp)


if __name__ == "__main__":
    main()