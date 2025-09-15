import os, math
import numpy as np
from scipy.signal import spectrogram
from lspopt import spectrogram_lspopt

def get_spectrogram(raw_signal, sampling_freq_hz:int, multi_taper:bool=False,
                    nperseg:int=128, noverlap:int=0):

    if multi_taper:
        f, t, Sxx = spectrogram_lspopt(
            raw_signal, fs=sampling_freq_hz, nperseg=nperseg, noverlap=noverlap, c_parameter=20.0
        )
    else:
        f, t, Sxx = spectrogram(
            raw_signal, fs=sampling_freq_hz, window="hann", nperseg=nperseg, noverlap=noverlap,
            detrend=False, scaling="density", mode="psd"
                    )
    
    return f, t, Sxx  # frequencies (F,), times (T,), spectrogram (F,T)



def write_spectrogram_dataset(results, out_dir, nperseg, noverlap,
                              multi_taper=False, fmin=0.5, fmax=None, eps=1e-10):
    os.makedirs(out_dir, exist_ok=True)

    # ---- Probe shapes from first item (with EXACT same params as compute) ----
    m0 = results[0]["meta"]
    sf = float(m0["sfreq"])
    n_ch = len(m0["ch_names"])
    T = int(m0["samples_per_epoch"])
    ny = sf / 2.0
    if fmax is None:
        fmax = ny

    x0 = np.zeros(T, np.float32)
    if multi_taper:
        f_probe, t_probe, S0 = spectrogram_lspopt(
            x0, fs=sf, nperseg=nperseg, noverlap=noverlap, c_parameter=20.0
        )
    else:
        f_probe, t_probe, S0 = spectrogram(
            x0, fs=sf, window="hann", nperseg=nperseg, noverlap=noverlap,
            detrend=False, scaling="density", mode="psd"
        )

    keep = (f_probe >= fmin) & (f_probe <= fmax)
    F, TT = int(keep.sum()), int(len(t_probe))

    # ---- Allocate memmaps ----
    N = sum(len(r["labels"]) for r in results)
    Xmm = np.memmap(os.path.join(out_dir, "spec_x.dat"), mode="w+",
                    dtype=np.float32, shape=(N, n_ch, F, TT))
    ymm = np.memmap(os.path.join(out_dir, "spec_y.dat"), mode="w+",
                    dtype=np.int64,   shape=(N,))

    # ---- Iterate and fill ----
    idx = 0
    for r in results:
        print(f"Processing {r.get('animal_id')}")
        X = r["epochs"].astype(np.float32, copy=False)  # (n_i, C, T)
        y = r["labels"].astype(np.int64, copy=False)
        n_i = len(y)

        for i in range(n_i):
            for c in range(n_ch):
                if multi_taper:
                    f, t, Sxx = spectrogram_lspopt(
                        X[i, c], fs=sf, nperseg=nperseg, noverlap=noverlap, c_parameter=20.0
                    )
                else:
                    f, t, Sxx = spectrogram(
                        X[i, c], fs=sf, window="hann", nperseg=nperseg, noverlap=noverlap,
                        detrend=False, scaling="density", mode="psd"
                    )
                # same keep mask as probe
                Xmm[idx + i, c] = np.log10(Sxx[keep] + eps)

        ymm[idx:idx + n_i] = y
        idx += n_i

    # ---- Save meta (from probe) ----
    meta = {
        "sfreq": sf,
        "channels": results[0]["meta"]["ch_names"],
        "method": "lspopt" if multi_taper else "stft",
        "multi_taper": bool(multi_taper),
        "c_parameter": 20.0 if multi_taper else None,
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "f": f_probe[keep].astype(np.float32),
        "t": t_probe.astype(np.float32),
        "shape": Xmm.shape,
    }
    np.savez(os.path.join(out_dir, "spec_meta.npz"), **meta)
    Xmm.flush(); ymm.flush()
    return str(Xmm.filename), str(ymm.filename), meta



# def write_spectrogram_dataset(results, out_dir, nperseg, noverlap, multi_taper=False, fmin=0.0, fmax=None, eps=1e-10):
#     os.makedirs(out_dir, exist_ok=True)
#     # probe shapes from first item
#     m0 = results[0]["meta"]
#     sf = int(m0["sfreq"])
#     n_ch = len(m0["ch_names"])
#     T = int(m0["samples_per_epoch"])
    
#     # Same parameters as Brodersen et al, 2024
#     if multi_taper:
#         f, t, Sxx = spectrogram_lspopt(np.zeros(T, np.float32), fs=sf, nperseg=sf, noverlap=None, c_parameter=20.0)
#     else:
#         f, t, Sxx = spectrogram(np.zeros(T, np.float32), fs=sf, window="hann",
#                             nperseg=nperseg, noverlap=noverlap,
#                             detrend=False, scaling="density", mode="psd")
        
#     keep = (f >= fmin) & (f <= (sf/2.0 if fmax is None else fmax))
#     F, TT = keep.sum(), len(t)

#     # total epochs
#     N = sum(len(r["labels"]) for r in results)

#     # memmaps
#     Xmm = np.memmap(os.path.join(out_dir, "spec_x.dat"), mode="w+", dtype=np.float32, shape=(N, n_ch, F, TT))
#     ymm = np.memmap(os.path.join(out_dir, "spec_y.dat"), mode="w+", dtype=np.int64, shape=(N,))

#     # iterate and fill
#     idx = 0
#     for r in results:
#         print(f"Processing {r.get('animal_id')}")
#         X = r["epochs"].astype(np.float32, copy=False)  # (n_i, C, T)
#         y = r["labels"].astype(np.int64, copy=False)
#         n_i = len(y)

#         # compute per-epoch/channel
#         for i in range(n_i):
#             for c in range(n_ch):
#                 if multi_taper:
#                     f, t, Sxx = spectrogram_lspopt(X[i, c], fs=sf, noverlap=None, c_parameter=20.0)
#                 else:
#                     f, t, Sxx = spectrogram(X[i, c], fs=sf, window="hann",
#                                         nperseg=nperseg, noverlap=noverlap,
#                                         detrend=False, scaling="density", mode="psd")
                    
#                 Xmm[idx + i, c] = np.log10(Sxx[keep] + eps)
#         ymm[idx:idx+n_i] = y
#         idx += n_i

#     meta = {
#         "sfreq": sf, "channels": results[0]["meta"]["ch_names"],
#         "nperseg": nperseg, "noverlap": noverlap,
#         "f": f[keep], "t": t, "shape": Xmm.shape,
#     }
#     np.savez(os.path.join(out_dir, "spec_meta.npz"), **meta)
#     # flush
#     Xmm.flush(); ymm.flush()
#     return str(Xmm.filename), str(ymm.filename), meta