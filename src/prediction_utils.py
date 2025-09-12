import numpy as np
from scipy.signal import medfilt

STAGE_TO_INT = {"Wake": 0, "NREM": 1, "REM": 2}
INT_TO_STAGE = {v:k for k,v in STAGE_TO_INT.items()}

def encode_stages(y):
    return np.array([STAGE_TO_INT.get(s, np.nan) for s in y], dtype=float)

def decode_stages(x):
    return np.array([INT_TO_STAGE.get(int(v), "Unknown") for v in x])

def smooth_median(y_pred, kernel=5):
    if kernel % 2 == 0:
        kernel += 1  # must be odd
    enc = encode_stages(y_pred).astype(int)
    sm = medfilt(enc, kernel_size=kernel)
    return decode_stages(sm)

def smooth_min_bout(y_pred, min_len={"REM":3, "NREM":2, "Wake":2}):
    y = np.array(y_pred).copy()
    runs, i = [], 0
    while i < len(y):
        j = i+1
        while j < len(y) and y[j] == y[i]: j += 1
        runs.append((i, j, y[i]))
        i = j
    for k, (s, e, lab) in enumerate(runs):
        if (e - s) < min_len.get(lab, 1):
            left_lab  = runs[k-1][2] if k > 0 else None
            right_lab = runs[k+1][2] if k < len(runs)-1 else None
            if left_lab is None and right_lab is None: continue
            if left_lab is None: new_lab = right_lab
            elif right_lab is None: new_lab = left_lab
            else:
                left_len  = runs[k-1][1] - runs[k-1][0]
                right_len = runs[k+1][1] - runs[k+1][0]
                new_lab = left_lab if left_len >= right_len else right_lab
            y[s:e] = new_lab
    return y

import numpy as np