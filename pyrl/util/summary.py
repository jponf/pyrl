# -*- coding: utf-8 -*-

"""Utility module that wraps tensorboard SummaryWriter and provides a
DummySummary that transforms all operations into no-ops.

For more information about the base SummaryWriter please see the
official PyTorch documentation:
https://pytorch.org/docs/stable/tensorboard.html
"""

import abc
from typing import Mapping, Optional, Union
import six

import torch.utils.tensorboard


###############################################################################


@six.add_metaclass(abc.ABCMeta)
class BaseSummary(object):
    """Interface that must be implemented by Summary objects."""

    @abc.abstractmethod
    def add_scalar(
        self,
        tag: str,
        scalar_value: Union[int, float],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ):
        """Add scalar data to summary.

        :param str tag: Data identifier.
        :param (int or float) scalar_value: Value to save.
        :param int global_step: Global step value to record.
        :param float walltime: Overrides default walltime (time.time()).
        """

    @abc.abstractmethod
    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Mapping[str, Union[int, float]],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ):
        """Adds many scalar data to summary.

        :param str main_tag: The parent name for the tags.
        :param dict tag_scalar_dict: Key-Value pair storing the tag and
            correspoding values.
        :param int global_step: Global step value to record.
        :param float walltime: Overrides default walltime (time.time()).
        """


class DummySummary(BaseSummary):
    """Dummy summary that transforms all operations into no-ops."""

    def add_scalar(
        self,
        tag: str,
        scalar_value: Union[int, float],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ):
        pass

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Mapping[str, Union[int, float]],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ):
        pass


class Summary(BaseSummary):
    """Summary class that calls the equivalent methods of the class
    `torch.util.tensorboard.SummaryWritter`
    """

    def __init__(
        self,
        log_dir: str,
        purge_step: Optional[int] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
    ):
        """Creates a `Summary` that will write out events and summaries
        to the event file.
        """
        self._writter = torch.utils.tensorboard.SummaryWriter(
            log_dir=log_dir,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )

    def add_scalar(
        self,
        tag: str,
        scalar_value: Union[int, float],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ):
        self._writter.add_scalar(
            tag=tag,
            scalar_value=scalar_value,
            global_step=global_step,
            walltime=walltime,
        )

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Mapping[str, Union[int, float]],
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ):
        self._writter.add_scalars(
            main_tag=main_tag,
            tag_scalar_dict=tag_scalar_dict,
            global_step=global_step,
            walltime=walltime,
        )
