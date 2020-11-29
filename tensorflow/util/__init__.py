from tensorflow import keras

import os
import time

# Keras Tensorboard Callback
def get_run_logdir(logpath=os.path.join(os.curdir, "tensorboard_logs")):
    run_id = time.strftime("run-%Y-%m-%d-%H-%M-%S")
    return os.path.join(logpath, run_id)


import numpy as np


def get_1cycle_schedule(
    lr_max=1e-3, n_data_points=8000, epochs=200, batch_size=40, verbose=0
):
    """
  Creates a look-up table of learning rates for 1cycle schedule with cosine annealing
  See @sgugger's & @jeremyhoward's code in fastai library: https://github.com/fastai/fastai/blob/master/fastai/train.py
  Wrote this to use with my Keras and (non-fastai-)PyTorch codes.
  Note that in Keras, the LearningRateScheduler callback (https://keras.io/callbacks/#learningratescheduler) only operates once per epoch, not per batch
      So see below for Keras callback

  Keyword arguments:
    lr_max            chosen by user after lr_finder
    n_data_points     data points per epoch (e.g. size of training set)
    epochs            number of epochs
    batch_size        batch size

  Output:
    lrs               look-up table of LR's, with length equal to total # of iterations

  Then you can use this in your PyTorch code by counting iteration number and setting
          optimizer.param_groups[0]['lr'] = lrs[iter_count]
  """
    if verbose > 0:
        print("Setting up 1Cycle LR schedule...")
    pct_start, div_factor = 0.3, 25.0  # @sgugger's parameters in fastai code
    lr_start = lr_max / div_factor
    lr_end = lr_start / 1e4
    n_iter = n_data_points * epochs // batch_size  # number of iterations
    a1 = int(n_iter * pct_start)
    a2 = n_iter - a1

    # make look-up table
    lrs_first = np.linspace(lr_start, lr_max, a1)  # linear growth
    lrs_second = (lr_max - lr_end) * (
        1 + np.cos(np.linspace(0, np.pi, a2))
    ) / 2 + lr_end  # cosine annealing
    lrs = np.concatenate((lrs_first, lrs_second))
    return lrs


class OneCycleScheduler(keras.callbacks.Callback):
    """My modification of Keras' Learning rate scheduler to do 1Cycle learning
       which increments per BATCH, not per epoch
    Keyword arguments
        **kwargs:  keyword arguments to pass to get_1cycle_schedule()
        Also, verbose: int. 0: quiet, 1: update messages.

    Sample usage (from my train.py):
        lrsched = OneCycleScheduler(lr_max=1e-4, n_data_points=X_train.shape[0], epochs=epochs, batch_size=batch_size, verbose=1)
    """

    def __init__(self, **kwargs):
        super(OneCycleScheduler, self).__init__()
        self.verbose = kwargs.get("verbose", 0)
        self.lrs = get_1cycle_schedule(**kwargs)
        self.iteration = 0

    def on_batch_begin(self, batch, logs=None):
        lr = self.lrs[self.iteration]
        keras.backend.set_value(
            self.model.optimizer.lr, lr
        )  # here's where the assignment takes place
        if self.verbose > 0:
            print(
                "\nIteration %06d: OneCycleScheduler setting learning "
                "rate to %s." % (self.iteration, lr)
            )
        self.iteration += 1

    def on_epoch_end(
        self, epoch, logs=None
    ):  # this is unchanged from Keras LearningRateScheduler
        logs = logs or {}
        logs["lr"] = keras.backend.get_value(self.model.optimizer.lr)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lrs = get_1cycle_schedule()
    epoch_list = np.linspace(0, epochs, len(lrs))
    plt.plot(epoch_list, lrs)
