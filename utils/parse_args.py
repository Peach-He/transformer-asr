import argparse


def parse_arguments():
    """Parse command-line arguments to the experiment.

    Arguments
    ---------
    arg_list : list, None
        A list of arguments to parse.  If not given, this is read from
        `sys.argv[1:]`

    Returns
    -------
    param_file : str
        The location of the parameters file.
    run_opts : dict
        Run options, such as distributed, device, etc.
    overrides : dict
        The overrides to pass to ``load_hyperpyyaml``.

    Example
    -------
    >>> argv = ['hyperparams.yaml', '--device', 'cuda:1', '--seed', '10']
    >>> filename, run_opts, overrides = parse_arguments(argv)
    >>> filename
    'hyperparams.yaml'
    >>> run_opts["device"]
    'cuda:1'
    >>> overrides
    'seed: 10'
    """
    parser = argparse.ArgumentParser(description="Transformer Training Reference")
    parser.add_argument(
        "--param_file",
        type=str,
        help="A yaml-formatted file using the extended YAML syntax. "
        "defined by SpeechBrain.",
    )
    parser.add_argument(
        "--log_config",
        type=str,
        help="A file storing the configuration options for logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to run the experiment on (e.g. 'cuda:0')",
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
        default="gloo",
        help="One of {nccl, gloo, mpi}",
    )
    parser.add_argument(
        "--find_unused_parameters",
        default=False,
        action="store_true",
        help="This flag disable unused parameters detection",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        help="Gradient norm will be clipped to this value, "
        "enter negative value to disable.",
    )
    parser.add_argument(
        "--grad_accumulation_factor",
        type=int,
        help="Number of batches to accumulate gradients before optimizer step",
    )

    return parser.parse_args()
