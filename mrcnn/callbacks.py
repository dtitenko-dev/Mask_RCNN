import os
import re
import six
import h5py
import json
import logging

import tensorflow.keras as keras

from tensorflow.python.keras import optimizers
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.distribute import distributed_file_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.training import checkpoint_management
from tensorflow.python.util import serialization


def save_optimizer_weights(model, filepath, overwrite=True, **kwargs):
    if not isinstance(filepath, h5py.File):
        # If file exists and should not be overwritten.
        if not overwrite and os.path.isfile(filepath):
            proceed = hdf5_format.ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return
        f = h5py.File(filepath, mode='w')
        opened_new_file = True
    else:
        f = filepath
        opened_new_file = False
    try:
        model_metadata = saving_utils.model_metadata(
            model, include_optimizer=True, require_config=False)
        for k, v in model_metadata.items():
            if isinstance(v, (dict, list, tuple)):
                f.attrs[k] = json.dumps(
                    v, default=serialization.get_json_type).encode('utf8')
            else:
                f.attrs[k] = v
        if not isinstance(model.optimizer, optimizers.TFOptimizer):
            hdf5_format.save_optimizer_weights_to_hdf5_group(f, model.optimizer)
        f.flush()
    finally:
        if opened_new_file:
            f.close()


def load_optimizer_weights(model, filepath):
    """Loads optimizer weights to compiled model from hdf5 file.
        Arguments:
            model: Compiled model
    """
    opened_new_file = not isinstance(filepath, h5py.File)
    if opened_new_file:
        f = h5py.File(filepath, mode='r')
    else:
        f = filepath

    try:
        if model.optimizer and 'optimizer_weights' in f:
            try:
                model.optimizer._create_all_weights(model.trainable_variables)
            except (NotImplementedError, AttributeError):
                logging.warning(
                    'Error when creating the weights of optimizer {}, making it '
                    'impossible to restore the saved optimizer state. As a result, '
                    'your model is starting with a freshly initialized optimizer.')
            optimizer_weight_values = hdf5_format.load_optimizer_weights_from_hdf5_group(f)
            try:
                model.optimizer.set_weights(optimizer_weight_values)
            except ValueError:
                logging.warning('Error in loading the saved optimizer '
                                'state. As a result, your model is '
                                'starting with a freshly initialized '
                                'optimizer.')
    finally:
        if opened_new_file:
            f.close()
    return model


class OptimizerCheckpoint(keras.callbacks.Callback):

    def __init__(self,
                 filepath,
                 verbose=0,
                 save_freq='epoch',
                 **kwargs):
        super(OptimizerCheckpoint, self).__init__()
        self.verbose = verbose
        self.filepath = path_to_string(filepath)
        self.save_freq = save_freq
        self.epochs_since_last_save = 0
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self._current_epoch = 0

        if 'load_weights_on_restart' in kwargs:
            self.load_weights_on_restart = kwargs['load_weights_on_restart']
            logging.warning('`load_weights_on_restart` argument is deprecated. '
                            'Please use `model.load_weights()` for loading weights '
                            'before the start of `model.fit()`.')
        else:
            self.load_weights_on_restart = False

        if 'period' in kwargs:
            self.period = kwargs['period']
            logging.warning('`period` argument is deprecated. Please use `save_freq` '
                            'to specify the frequency in number of batches seen.')
        else:
            self.period = 1

        if self.save_freq != 'epoch' and not isinstance(self.save_freq, int):
            raise ValueError('Unrecognized save_freq: {}'.format(self.save_freq))

        # Only the chief worker writes model checkpoints, but all workers
        # restore checkpoint at on_train_begin().
        self._chief_worker_only = False

    def on_train_begin(self, logs=None):
        if self.load_weights_on_restart:
            filepath_to_load = (
                self._get_most_recently_modified_file_matching_pattern(self.filepath))
            if (filepath_to_load is not None and
                    self._checkpoint_exists(filepath_to_load)):
                try:
                    # `filepath` may contain placeholders such as `{epoch:02d}`, and
                    # thus it attempts to load the most recently modified file with file
                    # name matching the pattern.
                    load_optimizer_weights(self.model, filepath=filepath_to_load)
                except (IOError, ValueError) as e:
                    raise ValueError('Error loading file from {}. Reason: {}'.format(
                        filepath_to_load, e))

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_optimizer_weights(epoch=self._current_epoch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.save_freq == 'epoch':
            self._save_optimizer_weights(epoch, logs)

    def _should_save_on_batch(self, batch):
        """Handles batch-level saving logic, supports steps_per_execution."""
        if self.save_freq == 'epoch':
            return False

        if batch <= self._last_batch_seen:  # New epoch.
            add_batches = batch + 1  # batches are zero-indexed.
        else:
            add_batches = batch - self._last_batch_seen
        self._batches_seen_since_last_saving += add_batches
        self._last_batch_seen = batch

        if self._batches_seen_since_last_saving >= self.save_freq:
            self._batches_seen_since_last_saving = 0
            return True
        return False

    def _save_optimizer_weights(self, epoch, logs=None):
        """Saves the optimizer weights.

        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}
        if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = tf_utils.to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)
            try:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                save_optimizer_weights(self.model, filepath, overwrite=True)
            except IOError as e:
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in six.ensure_str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for '
                                  'ModelCheckpoint. Filepath used is an existing '
                                  'directory: {}'.format(filepath))

    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
            # `filepath` may contain placeholders such as `{epoch:02d}` and
            # `{mape:.2f}`. A mismatch between logged metrics and the path's
            # placeholders can cause formatting to fail.
            file_path = self.filepath.format(epoch=epoch + 1, **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                           'Reason: {}'.format(self.filepath, e))
        self._write_filepath = distributed_file_utils.write_filepath(
            file_path, self.model.distribute_strategy)
        return self._write_filepath

    def _maybe_remove_file(self):
        # Remove the checkpoint directory in multi-worker training where this worker
        # should not checkpoint. It is a dummy directory previously saved for sync
        # distributed training.
        distributed_file_utils.remove_temp_dir_with_filepath(
            self._write_filepath, self.model.distribute_strategy)

    def _checkpoint_exists(self, filepath):
        """Returns whether the checkpoint `filepath` refers to exists."""
        if filepath.endswith('.h5'):
            return file_io.file_exists(filepath)
        tf_saved_optimizer_exists = file_io.file_exists(filepath + '.h5')
        return tf_saved_optimizer_exists

    def _get_most_recently_modified_file_matching_pattern(self, pattern):
        """Returns the most recently modified filepath matching pattern.

        Pattern may contain python formatting placeholder. If
        `tf.train.latest_checkpoint()` does not return None, use that; otherwise,
        check for most recently modified one that matches the pattern.

        In the rare case where there are more than one pattern-matching file having
        the same modified time that is most recent among all, return the filepath
        that is largest (by `>` operator, lexicographically using the numeric
        equivalents). This provides a tie-breaker when multiple files are most
        recent. Note that a larger `filepath` can sometimes indicate a later time of
        modification (for instance, when epoch/batch is used as formatting option),
        but not necessarily (when accuracy or loss is used). The tie-breaker is
        put in the logic as best effort to return the most recent, and to avoid
        undeterministic result.

        Modified time of a file is obtained with `os.path.getmtime()`.

        This utility function is best demonstrated via an example:

        ```python
        file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
        test_dir = self.get_temp_dir()
        path_pattern = os.path.join(test_dir, file_pattern)
        file_paths = [
            os.path.join(test_dir, file_name) for file_name in
            ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
        ]
        for file_path in file_paths:
          # Write something to each of the files
        self.assertEqual(
            _get_most_recently_modified_file_matching_pattern(path_pattern),
            file_paths[-1])
        ```

        Arguments:
            pattern: The file pattern that may optionally contain python placeholder
                such as `{epoch:02d}`.

        Returns:
            The most recently modified file's full filepath matching `pattern`. If
            `pattern` does not contain any placeholder, this returns the filepath
            that
            exactly matches `pattern`. Returns `None` if no match is found.
        """
        dir_name = os.path.dirname(pattern)
        base_name = os.path.basename(pattern)
        base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

        # If tf.train.latest_checkpoint tells us there exists a latest checkpoint,
        # use that as it is more robust than `os.path.getmtime()`.
        latest_tf_checkpoint = checkpoint_management.latest_checkpoint(dir_name)
        if latest_tf_checkpoint is not None and re.match(
                base_name_regex, os.path.basename(latest_tf_checkpoint)):
            return latest_tf_checkpoint

        latest_mod_time = 0
        file_path_with_latest_mod_time = None
        n_file_with_latest_mod_time = 0
        file_path_with_largest_file_name = None

        if file_io.file_exists(dir_name):
            for file_name in os.listdir(dir_name):
                # Only consider if `file_name` matches the pattern.
                if re.match(base_name_regex, file_name):
                    file_path = os.path.join(dir_name, file_name)
                    mod_time = os.path.getmtime(file_path)
                    if (file_path_with_largest_file_name is None or
                            file_path > file_path_with_largest_file_name):
                        file_path_with_largest_file_name = file_path
                    if mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                        file_path_with_latest_mod_time = file_path
                        # In the case a file with later modified time is found, reset
                        # the counter for the number of files with latest modified time.
                        n_file_with_latest_mod_time = 1
                    elif mod_time == latest_mod_time:
                        # In the case a file has modified time tied with the most recent,
                        # increment the counter for the number of files with latest modified
                        # time by 1.
                        n_file_with_latest_mod_time += 1

        if n_file_with_latest_mod_time == 1:
            # Return the sole file that has most recent modified time.
            return file_path_with_latest_mod_time
        else:
            # If there are more than one file having latest modified time, return
            # the file path with the largest file name.
            return file_path_with_largest_file_name

