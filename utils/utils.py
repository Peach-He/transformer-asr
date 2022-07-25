import torch
import torch.distributed as dist
import logging
import os
import sys
import shutil
import inspect
from hyperpyyaml import resolve_references
from datetime import date
import argparse

from utils.logger import setup_logging


logger = logging.getLogger(__name__)
DEFAULT_LOG_CONFIG = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_CONFIG = os.path.join(DEFAULT_LOG_CONFIG, "log-config.yaml")

def create_experiment_directory(
    experiment_directory,
    hyperparams_to_save=None,
    overrides={},
    log_config=DEFAULT_LOG_CONFIG
):
    try:
        # all writing command must be done with the main_process
        if is_main_process():
            if not os.path.isdir(experiment_directory):
                os.makedirs(experiment_directory)

            # Write the parameters file
            if hyperparams_to_save is not None:
                hyperparams_filename = os.path.join(
                    experiment_directory, "hyperparams.yaml"
                )
                with open(hyperparams_to_save) as f:
                    resolved_yaml = resolve_references(f, overrides)
                with open(hyperparams_filename, "w") as w:
                    print("# Generated %s from:" % date.today(), file=w)
                    print("# %s" % os.path.abspath(hyperparams_to_save), file=w)
                    print("# yamllint disable", file=w)
                    shutil.copyfileobj(resolved_yaml, w)

            # Copy executing file to output directory
            module = inspect.getmodule(inspect.currentframe().f_back)
            if module is not None:
                callingfile = os.path.realpath(module.__file__)
                shutil.copy(callingfile, experiment_directory)

            # Log exceptions to output automatically
            log_file = os.path.join(experiment_directory, "log.txt")
            logger_overrides = {
                "handlers": {"file_handler": {"filename": log_file}}
            }
            setup_logging(log_config, logger_overrides)
            sys.excepthook = _logging_excepthook

            # Log beginning of experiment!
            logger.info("Beginning experiment!")
            logger.info(f"Experiment folder: {experiment_directory}")

    finally:
        # wait for main_process if ddp is used
        ddp_barrier()

def _logging_excepthook(exc_type, exc_value, exc_traceback):
    """Interrupt exception raising to log the error."""
    logger.error("Exception:", exc_info=(exc_type, exc_value, exc_traceback))

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

def check_gradients(modules, loss, max_grad_norm, nonfinite_count, nonfinite_patience=3):
    if not torch.isfinite(loss):
        nonfinite_count += 1
        # Print helpful debug info
        logger.warn(f"Loss is {loss}.")
        for p in modules.parameters():
            if not torch.isfinite(p).all():
                logger.warn("Parameter is not finite: " + str(p))
        # Check if patience is exhausted
        if nonfinite_count > nonfinite_patience:
            raise ValueError(
                "Loss is not finite and patience is exhausted. "
                "To debug, wrap `fit()` with "
                "autograd's `detect_anomaly()`, e.g.\n\nwith "
                "torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
            )
        else:
            logger.warn("Patience not yet exhausted, ignoring this batch.")
            return False, nonfinite_count
    # Clip gradient norm
    torch.nn.utils.clip_grad_norm_(
        (p for p in modules.parameters()), max_grad_norm
    )
    return True, nonfinite_count

def update_average(loss, avg_loss, step):
    if torch.isfinite(loss):
        avg_loss -= avg_loss / step
        avg_loss += float(loss) / step
    return avg_loss


def parse_arguments(arg_list=None):
    if arg_list is None:
        arg_list = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Run a Transformer ASR experiment")
    parser.add_argument(
        "param_file",
        type=str,
        help="A yaml-formatted file using the extended YAML syntax. ",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Run the experiment with only a few batches for all "
        "datasets, to ensure code runs without crashing.",
    )
    parser.add_argument(
        "--debug_batches",
        type=int,
        default=2,
        help="Number of batches to run in debug mode.",
    )
    parser.add_argument(
        "--debug_epochs",
        type=int,
        default=2,
        help="Number of epochs to run in debug mode. "
        "If a non-positive number is passed, all epochs are run.",
    )
    parser.add_argument(
        "--log_config",
        type=str,
        help="A file storing the configuration options for logging",
    )
    # if use_env = False in torch.distributed.lunch then local_rank arg is given
    parser.add_argument("--local_rank", type=int, help="Rank on local machine")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device to run the experiment on (e.g. 'cuda:0')",
    )
    parser.add_argument(
        "--data_parallel_backend",
        default=False,
        action="store_true",
        help="This flag enables training with data_parallel.",
    )
    parser.add_argument(
        "--distributed_launch",
        default=False,
        action="store_true",
        help="This flag enables training with DDP. Assumes script run with "
        "`torch.distributed.launch`",
    )
    parser.add_argument(
        "--distributed_backend",
        type=str,
        default="nccl",
        help="One of {nccl, gloo, mpi}",
    )
    parser.add_argument(
        "--find_unused_parameters",
        default=False,
        action="store_true",
        help="This flag disable unused parameters detection",
    )
    parser.add_argument(
        "--jit_module_keys",
        type=str,
        nargs="*",
        help="A list of keys in the 'modules' dict to jitify",
    )
    parser.add_argument(
        "--auto_mix_prec",
        default=None,
        action="store_true",
        help="This flag enables training with automatic mixed-precision.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        help="Gradient norm will be clipped to this value, "
        "enter negative value to disable.",
    )
    parser.add_argument(
        "--nonfinite_patience",
        type=int,
        help="Max number of batches per epoch to skip if loss is nonfinite.",
    )
    parser.add_argument(
        "--noprogressbar",
        default=None,
        action="store_true",
        help="This flag disables the data loop progressbars.",
    )
    parser.add_argument(
        "--ckpt_interval_minutes",
        type=float,
        help="Amount of time between saving intra-epoch checkpoints "
        "in minutes. If non-positive, intra-epoch checkpoints are not saved.",
    )
    parser.add_argument(
        "--grad_accumulation_factor",
        type=int,
        help="Number of batches to accumulate gradients before optimizer step",
    )
    parser.add_argument(
        "--optimizer_step_limit",
        type=int,
        help="Number of optimizer steps to run. If not passed, all epochs are run.",
    )

    # Accept extra args to override yaml
    run_opts, overrides = parser.parse_known_args(arg_list)

    # Ignore items that are "None", they were not passed
    run_opts = {k: v for k, v in vars(run_opts).items() if v is not None}

    param_file = run_opts["param_file"]
    del run_opts["param_file"]

    overrides = _convert_to_yaml(overrides)

    # Checking that DataParallel use the right number of GPU
    if run_opts["data_parallel_backend"]:
        if torch.cuda.device_count() == 0:
            raise ValueError("You must have at least 1 GPU.")

    # For DDP, the device args must equal to local_rank used by
    # torch.distributed.launch. If run_opts["local_rank"] exists,
    # use os.environ["LOCAL_RANK"]
    local_rank = None
    if "local_rank" in run_opts:
        local_rank = run_opts["local_rank"]
    else:
        if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] != "":
            local_rank = int(os.environ["LOCAL_RANK"])

    # force device arg to be the same as local_rank from torch.distributed.lunch
    if local_rank is not None and "cuda" in run_opts["device"]:
        run_opts["device"] = run_opts["device"][:-1] + str(local_rank)

    return param_file, run_opts, overrides

def _convert_to_yaml(overrides):
    """Convert args to yaml for overrides"""
    yaml_string = ""

    # Handle '--arg=val' type args
    joined_args = "=".join(overrides)
    split_args = joined_args.split("=")

    for arg in split_args:
        if arg.startswith("--"):
            yaml_string += "\n" + arg[len("--") :] + ":"
        else:
            yaml_string += " " + arg

    return yaml_string.strip()
