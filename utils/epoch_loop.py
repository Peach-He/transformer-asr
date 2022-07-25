from utils.checkpoints import register_checkpoint_hooks
from utils.checkpoints import mark_as_saver
from utils.checkpoints import mark_as_loader
import logging


@register_checkpoint_hooks
class EpochCounter:
    """An epoch counter which can save and recall its state.

    Use this as the iterator for epochs.
    Note that this iterator gives you the numbers from [1 ... limit] not
    [0 ... limit-1] as range(limit) would.
    """

    def __init__(self, limit):
        self.current = 0
        self.limit = int(limit)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.limit:
            self.current += 1
            return self.current
        raise StopIteration

    @mark_as_saver
    def _save(self, path):
        with open(path, "w") as fo:
            fo.write(str(self.current))

    @mark_as_loader
    def _recover(self, path, end_of_epoch=True, device=None):
        # NOTE: end_of_epoch = True by default so that when
        #  loaded in parameter transfer, this starts a new epoch.
        #  However, parameter transfer to EpochCounter should
        #  probably never be used really.
        del device  # Not used.
        with open(path) as fi:
            saved_value = int(fi.read())
            if end_of_epoch:
                self.current = saved_value
            else:
                self.current = saved_value - 1
