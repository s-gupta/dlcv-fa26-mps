"""Plots a sweep for question 4.

Reads sweep.csv, so you can restyle the plot without re-running the sweep. Pass one
--run_dir per curve
"""

from absl import flags, app
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('run_dir', None, 'Directories holding sweep.csv, one per curve.')
flags.DEFINE_string('xlabel', 'number of training examples N', 'Label for the x axis.')
flags.DEFINE_string('out', 'sweep.png', 'Figure to write.')

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']


def main(_):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, run_dir in enumerate(FLAGS.run_dir):
        x, train, val, sem = np.loadtxt(f'{run_dir}/sweep.csv', delimiter=',', skiprows=1).T
        color = COLORS[i % len(COLORS)]
        label = run_dir.rstrip('/').split('/')[-1]
        ax.plot(x, val, '-o', color=color, ms=4, lw=2, label=f'{label}: val')
        ax.fill_between(x, val - sem, val + sem, color=color, alpha=0.2, lw=0)
        ax.plot(x, train, '--o', color=color, ms=3, lw=1.5, alpha=0.45,
                label=f'{label}: train')

    ax.set_xscale('log')
    ax.set_xlabel(FLAGS.xlabel)
    ax.set_ylabel('0-1 error')
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FLAGS.out, dpi=130)
    print(f'wrote {FLAGS.out}')


if __name__ == '__main__':
    app.run(main)
