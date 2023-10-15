# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10_concepts.ipynb.

# %% auto 0
__all__ = ['get_window']

# %% ../nbs/10_concepts.ipynb 2
import numpy as np


# %% ../nbs/10_concepts.ipynb 8
def get_window(x, window_size=10, end_step=100, indices=None, return_indices=True):
    """
    Returns a window from x of window_size, ending in end_step.

    If actual indices are passed, a window corresponding to that will be taken.
    """
    start_step = end_step - window_size
    indices = np.asarray(range(0, len(x))) if indices is None else indices
    if return_indices:
        return indices[start_step:end_step]
    else:
        return x[
            indices[start_step:end_step], :
        ]  # x of shape (num_features, feature_len)

