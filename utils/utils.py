import torch
import torch.distributed as dist
import logging


logger = logging.getLogger(__name__)

def run_on_main(
    func,
    args=None,
    kwargs=None,
    post_func=None,
    post_args=None,
    post_kwargs=None,
    run_post_on_main=False,
):
    # Handle the mutable data types' default args:
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if post_args is None:
        post_args = []
    if post_kwargs is None:
        post_kwargs = {}

    if is_main_process():
        # Main comes here
        try:
            func(*args, **kwargs)
        finally:
            ddp_barrier()
    else:
        # Others go here
        ddp_barrier()
    if post_func is not None:
        if run_post_on_main:
            # Just run on every process without any barrier.
            post_func(*post_args, **post_kwargs)
        elif not is_main_process():
            # Others go here
            try:
                post_func(*post_args, **post_kwargs)
            finally:
                ddp_barrier()
        else:
            # But main comes here
            ddp_barrier()

def is_main_process():
    if not dist.is_initialized() or dist.get_rank() == 0:
        return True
    else:
        return False

def ddp_barrier():
    if dist.is_initialized():
        dist.barrier()

def check_gradients(loss):
    if not torch.isfinite(loss):
        logger.warn(f"Loss is {loss}.")
        # Check if patience is exhausted
        raise ValueError(
            "Loss is not finite and patience is exhausted. "
            "To debug, wrap `fit()` with "
            "autograd's `detect_anomaly()`, e.g.\n\nwith "
            "torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
        )
    return True

def update_average(loss, avg_loss, step):
    if torch.isfinite(loss):
        avg_loss -= avg_loss / step
        avg_loss += float(loss) / step
    return avg_loss