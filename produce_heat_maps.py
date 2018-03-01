import matplotlib
matplotlib.use('macosx')
import pickle
import seaborn
import matplotlib.pyplot as plt

EXPERIMENT_NAME = 'copy_ntm'
MANN = 'NTM'
TASK = 'Copy'

HEAD_LOG_FILE = 'head_logs/{0}.p'.format(EXPERIMENT_NAME)
GENERALIZATION_HEAD_LOG_FILE = 'head_logs/generalization_{0}.p'.format(EXPERIMENT_NAME)

outputs = pickle.load(open(HEAD_LOG_FILE, "rb"))
outputs.update(pickle.load(open(GENERALIZATION_HEAD_LOG_FILE, "rb")))

def plot_figures(figures, nrows=1, ncols=1, width_ratios=None):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, gridspec_kw={'width_ratios': width_ratios})

    for ind, (title, fig) in enumerate(figures):
        axeslist.ravel()[ind].imshow(fig, cmap='gray', interpolation='nearest')
        axeslist.ravel()[ind].set_title(title)
        if TASK != 'Associative Recall' or ind == 0:
            axeslist.ravel()[ind].set_xlabel('Time ------->')
    
    if TASK == 'Associative Recall':
        plt.sca(axeslist[1])
        plt.xticks([0, 1, 2])
        plt.sca(axeslist[2])
        plt.xticks([0, 1, 2])

    if TASK == 'Copy':
        plt.sca(axeslist[1])
        plt.yticks([])

    plt.tight_layout()

for seq_len, heat_maps_list in outputs.iteritems():
    for step, heat_maps in enumerate(heat_maps_list[-2:] if len(heat_maps_list) >= 2 else heat_maps_list):
        inputs = heat_maps['inputs'].T
        labels = heat_maps['labels'].T
        outputs = heat_maps['outputs'].T

        if TASK == 'Copy':
            plot_figures([('{0} - {1} - Inputs'.format(MANN, TASK), inputs), ('Outputs', outputs)], 1, 2, width_ratios=[2, 1.1])
            plt.savefig('head_logs/img/{0}_{1}_{2}'.format(EXPERIMENT_NAME, seq_len, step), bbox_inches='tight')
            plt.close()
        elif TASK == 'Associative Recall':
            plot_figures([('{0} - {1} - Inputs'.format(MANN, TASK), inputs), ('Labels', labels), ('Outputs', outputs)], 1, 3, width_ratios=[seq_len+2, 1, 1])
            plt.savefig('head_logs/img/{0}_{1}_{2}'.format(EXPERIMENT_NAME, seq_len, step), bbox_inches='tight')
            plt.close()
