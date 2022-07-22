import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import _BaseDataLoaderIter
import logging
from pathlib import Path
import warnings
import functools
from torch.utils.data import DistributedSampler
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.sampler import ReproducibleRandomSampler
from speechbrain.dataio.sampler import DistributedSamplerWrapper
from speechbrain.dataio.sampler import ReproducibleRandomSampler
from speechbrain.utils.checkpoints import (
    register_checkpoint_hooks,
    mark_as_saver,
    mark_as_loader,
)


logger = logging.getLogger(__name__)

def make_dataloader(dataset, stage, dist, **loader_kwargs):
    if stage == 'train':
        loader_kwargs = train_loader_specifics(dataset, dist, loader_kwargs)

    # PaddedBatch as default collation for DynamicItemDataset
    if "collate_fn" not in loader_kwargs and isinstance(
        dataset, DynamicItemDataset
    ):
        loader_kwargs["collate_fn"] = PaddedBatch
    # Reproducible random sampling
    if loader_kwargs.get("shuffle", False):
        if loader_kwargs.get("sampler") is not None:
            raise ValueError(
                "Cannot specify both shuffle=True and a "
                "sampler in loader_kwargs"
            )
        sampler = ReproducibleRandomSampler(dataset)
        loader_kwargs["sampler"] = sampler
        # Should delete shuffle because you can't set both Sampler and
        # shuffle
        # NOTE: the dict of loader options may get used elsewhere!
        # However, this del doesn't touch those because loader_kwargs comes
        # from a **kwargs dict.
        del loader_kwargs["shuffle"]
    # Create the loader
    if isinstance(dataset, IterableDataset):
        dataloader = DataLoader(dataset, **loader_kwargs)
    else:
        dataloader = SaveableDataLoader(dataset, **loader_kwargs)
    return dataloader


def train_loader_specifics(dataset, distributed_launch, loader_kwargs):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    sampler = loader_kwargs.get("sampler", None)
    # Shuffling should really only matter for the train stage. Shuffling
    # will also lead to more padding in batches if the order was otherwise
    # sorted by length.
    shuffle = loader_kwargs.get("shuffle", False)
    if shuffle and not distributed_launch:
        if sampler is not None:
            raise ValueError(
                "Cannot specify both shuffle=True"
                "and a sampler in loader_kwargs"
            )
        sampler = ReproducibleRandomSampler(dataset)
        train_sampler = sampler
        loader_kwargs["sampler"] = train_sampler
        # Delete the shuffle flag, since you cannot specify both a sampler and
        # shuffling:
        del loader_kwargs["shuffle"]
    # Possibly make a DistributedSampler or a wrapper for some other sampler
    if distributed_launch and not isinstance(dataset, IterableDataset):
        drop_last = loader_kwargs.get("drop_last", False)
        # num_replicas arg is equal to world_size
        # and retrieved automatically within
        # DistributedSampler obj.
        if sampler is not None:
            train_sampler = DistributedSamplerWrapper(
                sampler,
                rank=rank,
                drop_last=drop_last,
                shuffle=shuffle,
            )
            # with DistributedSamplerWrapper, one must disable shuffling for dataloader
            loader_kwargs["shuffle"] = False
            loader_kwargs["sampler"] = train_sampler
        elif loader_kwargs.get("batch_sampler") is None:
            # no sampler and batch-sampler
            train_sampler = DistributedSampler(
                dataset, rank=rank, shuffle=False, drop_last=drop_last
            )
            # with DistributedSamplerWrapper, one must disable shuffling for dataloader
            loader_kwargs["shuffle"] = False
            loader_kwargs["sampler"] = train_sampler
        else:  # batch_sampler was specified
            train_sampler = DistributedSamplerWrapper(
                loader_kwargs.get("batch_sampler", None),
                rank=rank,
                shuffle=False,
            )
            loader_kwargs["batch_sampler"] = train_sampler
    elif distributed_launch and isinstance(dataset, IterableDataset):
        logger.warning(
            "Cannot automatically solve distributed sampling "
            "for IterableDataset."
        )
    return loader_kwargs

class SaveableDataLoader(DataLoader):
    """A saveable version of the PyTorch DataLoader.

    See `torch.utils.data.DataLoader` for usage. This class should work exactly
    like the PyTorch basic DataLoader, but this can be checkpointed with
    SpeechBrain's Checkpointer.

    Note
    ----
    1. The saveability is implemented via some unfortunately slightly magical
    means.
    2. The data loader cannot recover after entering __iter__. Normally this is
    not a problem, as recovery should happen before training begins.  However,
    just before evaluation, it is also typical to recover the checkpoint at
    which performance was the best. Thus, if a checkpoint is loaded after
    entering __iter__, we just assume it is for this reason. A warning is
    logged, but that is all.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "SaveableDataLoader cannot save the position in an "
                "IterableDataset. Save the position on the dataset itself."
            )
        self._speechbrain_recovery_skip_to = None
        self._speechbrain_iterator = None

    def __iter__(self):
        iterator = super().__iter__()
        # Keep a reference to the iterator,
        # to be able to access the iterator._num_yielded value.
        # Keep a full reference (keeping the iterator alive)
        # rather than e.g. a weakref, as we may want to save a checkpoint
        # after the iterator has been exhausted, but before the full epoch has
        # ended (e.g. validation is still running)
        self._speechbrain_iterator = iterator
        return iterator

    @mark_as_saver
    def _speechbrain_save(self, path):
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "Warning again: a checkpoint was requested on "
                "SaveableDataLoader, but the dataset is an IterableDataset. "
                "Cannot save the position in an IterableDataset. Not raising "
                "an error; assuming that you know what you're doing."
            )
        if self._speechbrain_iterator is None:
            to_save = None
        else:
            to_save = self._speechbrain_iterator._num_yielded
        with open(path, "w") as fo:
            fo.write(str(to_save))

    @mark_as_loader
    def _speechbrain_load(self, path, end_of_epoch, device=None):
        del device  # Unused here
        if self._speechbrain_iterator is not None:
            logging.debug(
                "SaveableDataLoader was requested to load a "
                "checkpoint, but the DataLoader has already been "
                "iterated. The DataLoader file will be ignored. "
                "This is normal in evaluation, when a checkpoint is "
                "loaded just to retrieve the best model."
            )
            return
        if end_of_epoch:
            # Don't load at end of epoch, as we actually want to start a fresh
            # epoch iteration next.
            return
        with open(path) as fi:
            saved = fi.read()
            if saved == str(None):
                # Saved at a point where e.g. an iterator did not yet exist.
                return
            else:
                self._speechbrain_recovery_skip_to = int(saved)