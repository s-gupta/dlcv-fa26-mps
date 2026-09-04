"""Sweeps the training set size or the feature count and records train and val error.
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import logging

from absl import flags, app
import numpy as np

from featurize import featurize
from models import LinearRegressionClassifier
from utils import get_mnist_dataset, compute_accuracy, get_noisy_mnist

FLAGS = flags.FLAGS

# Runs averaged at each sweep point. A single run is noisy enough to hide the trend.

flags.DEFINE_enum('sweep', 'samples', ['samples', 'features'],
                  'Vary the training set size or the number of random projections.')
flags.DEFINE_integer('num_train', 800, 'Training set size, when sweeping features.')
flags.DEFINE_integer('trials', 5, 'Number of trials to average over at each sweep point.')
flags.DEFINE_integer('num_features', 800, 'Feature count, when sweeping samples.')
flags.DEFINE_integer('sweep_min', 10, 'Smallest value of the swept quantity.')
flags.DEFINE_integer('sweep_max', 2000, 'Largest value of the swept quantity.')
flags.DEFINE_integer('sweep_count', 12, 'How many values to sweep, spaced geometrically.')
flags.DEFINE_float('label_noise', 0.05, 'Fraction of training labels to flip.')
flags.DEFINE_float('reg_wt', 1e-5, 'Weight of the regularization term in the loss function.')
flags.DEFINE_string('out_dir', 'runs/sweep', 'Where to store results.')

def run_setting(num_train, num_features, trial):
    data_val, labels_val = get_mnist_dataset('val')
    data_val = featurize(data_val, 'rrp', num_rrp=num_features, seed=trial)

    data_train, labels_train = get_noisy_mnist('train', num_train, 
                                               noise_level=FLAGS.label_noise, 
                                               noise_seed=trial, 
                                               training_shuffle_seed=trial)
    data_train = featurize(data_train, 'rrp', num_rrp=num_features, seed=trial)

    model = LinearRegressionClassifier(data_train, labels_train, reg_wt=FLAGS.reg_wt)
    model.train()
    preds_train = model.predict(data_train)
    preds_val = model.predict(data_val)
    train_accuracy = compute_accuracy(labels_train, preds_train)
    val_accuracy = compute_accuracy(labels_val, preds_val)
    train_err = 1-train_accuracy
    val_err = 1-val_accuracy
    return train_err, val_err

def main(_):
    # geometrically spaced, so the sweep is evenly sampled on the log axis it is
    # plotted on
    sweep = np.unique(np.geomspace(FLAGS.sweep_min, FLAGS.sweep_max,
                                   FLAGS.sweep_count).round().astype(int))
    logging.info(f'sweeping {FLAGS.sweep} over {list(sweep)}')
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    rows = []
    for value in sweep:
        num_train = value if FLAGS.sweep == 'samples' else FLAGS.num_train
        num_features = value if FLAGS.sweep == 'features' else FLAGS.num_features

        train_errs, val_errs = [], []
        for trial in range(FLAGS.trials):
            train_err, val_err = run_setting(num_train, num_features, trial)
            train_errs.append(train_err)
            val_errs.append(val_err)
        
        rows.append((value, np.mean(train_errs), np.mean(val_errs),
                     np.std(val_errs) / np.sqrt(len(val_errs))))
        logging.info(f'{FLAGS.sweep}: {value}, train_error: {rows[-1][1]:.4f}, '
                     f'val_error: {rows[-1][2]:.4f}')

    out = os.path.join(FLAGS.out_dir, 'sweep.csv')
    np.savetxt(out, np.array(rows), delimiter=',', comments='',
               header=f'{FLAGS.sweep},train_error,val_error,val_error_sem')
    logging.info(f'wrote {out}')

if __name__ == '__main__':
    app.run(main)
