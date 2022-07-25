import os
import yaml
import tqdm
import logging
import logging.config
import math
import collections.abc

ORDERS_ABBREV = {
    -24: "y",
    -21: "z",
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    21: "Z",
    24: "Y",
}

# Short scale
# Negative powers of ten in lowercase, positive in uppercase
ORDERS_WORDS = {
    -24: "septillionths",
    -21: "sextillionths",
    -18: "quintillionths",
    -15: "quadrillionths",
    -12: "trillionths",
    -9: "billionths",
    -6: "millionths",
    -3: "thousandths",
    0: "",
    3: "Thousand",
    6: "Million",
    9: "Billion",
    12: "Trillion",
    15: "Quadrillion",
    18: "Quintillion",
    21: "Sextillion",
    24: "Septillion",
}


def setup_logging(
    config_path="log-config.yaml", overrides={}, default_level=logging.INFO,
):
    """Setup logging configuration.

    Arguments
    ---------
    config_path : str
        The path to a logging config file.
    default_level : int
        The level to use if the config file is not found.
    overrides : dict
        A dictionary of the same structure as the config dict
        with any updated values that need to be applied.
    """
    if os.path.exists(config_path):
        with open(config_path, "rt") as f:
            config = yaml.safe_load(f)
        recursive_update(config, overrides)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def recursive_update(d, u, must_match=False):
    """Similar function to `dict.update`, but for a nested `dict`.

    From: https://stackoverflow.com/a/3233356

    If you have to a nested mapping structure, for example:

        {"a": 1, "b": {"c": 2}}

    Say you want to update the above structure with:

        {"b": {"d": 3}}

    This function will produce:

        {"a": 1, "b": {"c": 2, "d": 3}}

    Instead of:

        {"a": 1, "b": {"d": 3}}

    Arguments
    ---------
    d : dict
        Mapping to be updated.
    u : dict
        Mapping to update with.
    must_match : bool
        Whether to throw an error if the key in `u` does not exist in `d`.
    """
    # TODO: Consider cases where u has branch off k, but d does not.
    # e.g. d = {"a":1}, u = {"a": {"b": 2 }}
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping) and k in d:
            recursive_update(d.get(k, {}), v)
        elif must_match and k not in d:
            raise KeyError(
                f"Override '{k}' not found in: {[key for key in d.keys()]}"
            )
        else:
            d[k] = v
