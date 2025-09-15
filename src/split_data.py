import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def make_group_splits(y, groups, n_splits=5, random_state=0, test_fold=0, val_fold=1):
    """
    y:        (N,) int labels
    groups:   (N,) array-like of animal IDs (strings or ints)
    Returns:  train_idx, val_idx, test_idx (disjoint index arrays)
    """
    y = np.asarray(y)
    groups = np.asarray(groups)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(sgkf.split(np.zeros_like(y), y, groups))  # each item: (train_idx, test_idx_for_that_fold)

    idx_test = folds[test_fold][1]
    idx_val  = folds[val_fold][1]
    all_idx  = np.arange(len(y))
    idx_train = np.setdiff1d(all_idx, np.union1d(idx_test, idx_val), assume_unique=False)

    # sanity: no animal appears in more than one split
    assert set(groups[idx_train]).isdisjoint(groups[idx_val])
    assert set(groups[idx_train]).isdisjoint(groups[idx_test])
    assert set(groups[idx_val]).isdisjoint(groups[idx_test])

    return idx_train, idx_val, idx_test