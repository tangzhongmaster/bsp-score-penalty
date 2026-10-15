import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_integer("slices", 1, "Number of random vectors used in Hutchinson's trace estimator (K).")
flags.DEFINE_integer("bs", 128, "Batch size (b).")
flags.DEFINE_float("balancing_fac", 0.0001, "Balancing factor (lambda).")
flags.DEFINE_string("restore", None, "The path to a checkpoint file.")
flags.DEFINE_enum("reg_type", None, ["lqc_est", "bsp"],
                   "Which regularizer to use when balancing_fac>0. Defaults to whatever the "
                   "config file specifies (usually 'lqc_est'); pass --reg_type=bsp to fine-tune "
                   "with the Bilinear Symmetry Penalty instead.")
flags.DEFINE_boolean("bsp_std_weighting", None,
                      "Only relevant when reg_type=bsp. Multiply the penalty by the squared noise level. "
                      "Defaults to the config file's value.")
flags.DEFINE_integer("n_iters", None, "Override config.training.n_iters.")
flags.DEFINE_integer("snapshot_freq", None,
                      "Override config.training.snapshot_freq (checkpoint frequency).")
flags.DEFINE_integer("log_freq", None, "Override config.training.log_freq (loss logging frequency).")
flags.mark_flags_as_required(["workdir", "config"])

def main(argv):
  FLAGS.workdir = os.path.join("results", FLAGS.workdir)
  os.makedirs(FLAGS.workdir, exist_ok=True)
  FLAGS.config.loss.slices = FLAGS.slices
  FLAGS.config.loss.balancing_fac = FLAGS.balancing_fac
  FLAGS.config.training.batch_size = FLAGS.bs
  FLAGS.config.eval.batch_size = FLAGS.bs
  FLAGS.config.training.restore_path = FLAGS.restore
  if FLAGS.reg_type is not None:
    FLAGS.config.loss.reg_type = FLAGS.reg_type
  if FLAGS.bsp_std_weighting is not None:
    FLAGS.config.loss.bsp_std_weighting = FLAGS.bsp_std_weighting
  if FLAGS.n_iters is not None:
    FLAGS.config.training.n_iters = FLAGS.n_iters
  if FLAGS.snapshot_freq is not None:
    FLAGS.config.training.snapshot_freq = FLAGS.snapshot_freq
    FLAGS.config.training.snapshot_freq_for_preemption = FLAGS.snapshot_freq
  if FLAGS.log_freq is not None:
    FLAGS.config.training.log_freq = FLAGS.log_freq
  gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
  handler = logging.StreamHandler(gfile_stream)
  formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
  handler.setFormatter(formatter)
  logger = logging.getLogger()
  logger.addHandler(handler)
  logger.setLevel('INFO')
  run_lib.train(FLAGS.config, FLAGS.workdir)

if __name__ == "__main__":
  app.run(main)
