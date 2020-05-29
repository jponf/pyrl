# -*- coding: utf-8 -*-

"""Utility module that wraps tensorboard SummaryWriter and provides a
DummySummary that transforms all operations into no-ops.

For more information about the base SummaryWriter please see the
official PyTorch documentation:
https://pytorch.org/docs/stable/tensorboard.html
"""

import abc
import six

import torch.utils.tensorboard


###############################################################################

@six.add_metaclass(abc.ABCMeta)
class BaseSummary(object):
    """Interface that must be implemented by Summary objects."""

    @abc.abstractmethod
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        """Add scalar data to summary.

        :param str tag: Data identifier.
        :param (float or str) scalar_value: Value to save.
        :param int global_step: Global step value to record.
        :param float walltime: Overrides default walltime (time.time()).
        """

    @abc.abstractmethod
    def add_scalars(self, main_tag, tag_scalar_dict,
                    global_step=None, walltime=None):
        """Adds many scalar data to summary.

        :param str main_tag: The parent name for the tags.
        :param dict tag_scalar_dict: Key-Value pair storing the tag and
            correspoding values.
        :param int global_step: Global step value to record.
        :param float walltime: Overrides default walltime (time.time()).
        """


class DummySummary(BaseSummary):
    """Dummy summary that transforms all operations into no-ops."""

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass

    def add_scalars(self, main_tag, tag_scalar_dict,
                    global_step=None, walltime=None):
        pass


class Summary(BaseSummary):
    """Summary class that calls the equivalent methods of the class
    `torch.util.tensorboard.SummaryWritter`
    """

    def __init__(self, log_dir, purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix=''):
        """Creates a `Summary` that will write out events and summaries
        to the event file.
        """
        self._writter = torch.utils.SummaryWriter(
            log_dir=log_dir, purge_step=purge_step, max_queue=max_queue,
            flush_secs=flush_secs, filename_suffix=filename_suffix)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self._writter(tag=tag, scalar_value=scalar_value,
                      global_step=global_step, walltime=walltime)

    def add_scalars(self, main_tag, tag_scalar_dict,
                    global_step=None, walltime=None):
        self._writter(main_Tag=main_tag, tag_scalar_dict=tag_scalar_dict,
                      global_step=global_step, walltime=walltime)
