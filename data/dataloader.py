import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import _BaseDataLoaderIter
import logging
from pathlib import Path
import warnings
import functools
from dataio.batch import PaddedBatch, BatchsizeGuesser
import dataio.dataset
from dataio.dataset import DynamicItemDataset
from dataio.sampler import ReproducibleRandomSampler, DistributedSampler, DistributedSamplerWrapper
from utils.checkpoints import (
    register_checkpoint_hooks,
    mark_as_saver,
    mark_as_loader,
)
import utils.data_pipeline
import dataio.dataio
import processing
import processing.speech_augmentation


logger = logging.getLogger(__name__)

def make_dataloader(dataset, stage, dist, looped_nominal_epoch=None, **loader_kwargs):
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
    if looped_nominal_epoch is not None:
        dataloader = LoopedLoader(dataloader, looped_nominal_epoch)
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


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

#############################remove train dataset sorting
    valid_data = DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    valtest_datasets = [valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @utils.data_pipeline.takes("wav")
    @utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = dataio.dataio.read_audio(wav)
        return sig

    dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @utils.data_pipeline.takes("wav")
    @utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if hparams["speed_perturb"]:
            sig = dataio.dataio.read_audio(wav)
            # factor = np.random.uniform(0.95, 1.05)
            # sig = resample(sig.numpy(), 16000, int(16000*factor))
            speed = processing.speech_augmentation.SpeedPerturb(
                16000, [x for x in range(95, 105)]
            )
            sig = speed(sig.unsqueeze(0)).squeeze(0)  # torch.from_numpy(sig)
        else:
            sig = dataio.dataio.read_audio(wav)
        return sig

    dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    # 3. Define text pipeline:
    @utils.data_pipeline.takes("wrd")
    @utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )

    ########################remove batch sampler
    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
    )


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


@register_checkpoint_hooks
class LoopedLoader:
    """Loops an underlying iterable indefinitely, with nominal epoch lengths

    This is useful for working with IterableDatasets, and particularly
    webdataset-style loading. We recommend using ``.repeat()`` on the
    webdataset IterableDataset instance, so that the underlying dataloader
    naturally continues for ever.

    Arguments
    ---------
    loader : iterable
        A DataLoader or other iterable that is looped repeatedly.
    epoch_length : int
        The length of the nominal epoch. After this many steps, raises
        StopIteration
    """

    def __init__(self, loader, epoch_length, batchsize_fn=None):
        self.loader = loader
        self.iterator = None
        self.epoch_length = epoch_length
        self.step = 0  # Step in epoch
        self.total_steps = 0  # Total steps ever
        self.total_samples = 0  # Total samples seen on this process
        if batchsize_fn is None:
            self.batchsize_fn = BatchsizeGuesser()

    def __iter__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        return self

    def __next__(self):
        if self.step < self.epoch_length:
            self.step += 1
            self.total_steps += 1
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                batch = next(self.iterator)
            self.total_samples += self.batchsize_fn(batch)
            return batch
        else:
            self.step = 0
            raise StopIteration

    def __len__(self):
        return self.epoch_length

    @mark_as_saver
    def save(self, path):
        """Saves the needed information."""
        with open(path, "w") as fo:
            print(self.step, file=fo)
            print(self.total_steps, file=fo)
            print(self.total_samples, file=fo)

    @mark_as_loader
    def load(self, path, end_of_epoch=True, device=None):
        """Loads the needed information."""
        del device  # Unused here
        with open(path) as fi:
            self.step = int(fi.readline().strip())
            self.total_steps = int(fi.readline().strip())
            self.total_samples = int(fi.readline().strip())
            if not end_of_epoch and self.step == 0 and self.total_steps > 0:
                # Step has been set to 0 at the end of iteration,
                # so return it to epoch_length, so that first iteration
                # of this will immediately raise StopIteration.
                # Basically, this can happen when e.g. the main training
                # loop has already finished but there is a checkpoint in the
                # middle of validation.
                self.step = self.epoch_length