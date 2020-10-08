import os
import time

# Keras Tensorboard Callback
def get_run_logdir(logpath=os.path.join(os.curdir, "tensorboard_logs")):
    run_id = time.strftime("run-%Y-%m-%d-%H-%M-%S")
    return os.path.join(logpath, run_id)
