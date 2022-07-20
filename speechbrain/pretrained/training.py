"""
Training utilities for pretrained models

Authors
* Artem Ploujnikov 2021
"""
import os
import shutil


def save_for_pretrained(
    hparams,
    min_key=None,
    max_key=None,
    pretrainer_key="pretrainer",
    checkpointer_key="checkpointer",
):
    """
    Saves the necessary files for the pretrained model
    from the best checkpoint found. The goal of this function
    is to export the model for a Pretrainer

    Arguments
    ---------
    hparams: dict
        the hyperparameter file
    max_key : str
        Key to use for finding best checkpoint (higher is better).
        By default, passed to ``self.checkpointer.recover_if_possible()``.
    min_key : str
        Key to use for finding best checkpoint (lower is better).
        By def  ault, passed to ``self.checkpointer.recover_if_possible()``.
    checkpointer_key: str
        the key under which the checkpointer is stored
    pretrained_key: str
        the key under which the pretrainer is stored
    """
    if any(key not in hparams for key in [pretrainer_key, checkpointer_key]):
        raise ValueError(
            f"Incompatible hparams: a checkpointer with key {checkpointer_key}"
            f"and a pretrainer with key {pretrainer_key} are required"
        )
    pretrainer = hparams[pretrainer_key]
    checkpointer = hparams[checkpointer_key]
    checkpoint = checkpointer.find_checkpoint(min_key=min_key, max_key=max_key)

    pretrainer_keys = set(pretrainer.loadables.keys())
    checkpointer_keys = set(checkpoint.paramfiles.keys())
    keys_to_save = pretrainer_keys & checkpointer_keys
    for key in keys_to_save:
        source_path = checkpoint.paramfiles[key]
        if not os.path.exists(source_path):
            raise ValueError(
                f"File {source_path} does not exist in the checkpoint"
            )
        target_path = pretrainer.paths[key]
        dirname = os.path.dirname(target_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if os.path.exists(target_path):
            os.remove(target_path)
        shutil.copyfile(source_path, target_path)
