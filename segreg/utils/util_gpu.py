import numpy as np
import torch
import torch.fft
import torchvision.transforms.functional as TF

## tensor ##
VALUE_TYPE = torch.float32


def compute_entropy(C, N, eps=1e-7):
    """
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """
    p = C / N
    return p * torch.log2(torch.clamp(p, min=eps, max=None))


def corr_apply(A_fft, B_fft, sz, do_rounding=True):
    """
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """
    C_fft = fftconv(A_fft, B_fft)
    C = ifft(C_fft)
    C = C[: sz[0], : sz[1], : sz[2], : sz[3]]
    if do_rounding:
        C = torch.round(C)
    return C


def corr_target_setup(A):
    """
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """
    B = fft(A)
    return B


def corr_template_setup(B):
    """
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """
    B_fft = torch.conj(fft(B))
    return B_fft


def create_float_tensor(shape, on_gpu, fill_value=None):
    """
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """

    if on_gpu:
        res = torch.cuda.FloatTensor(shape[0], shape[1], shape[2], shape[3])
        if fill_value is not None:
            res.fill_(fill_value)
        return res
    else:
        if fill_value is not None:
            res = np.full(
                (shape[0], shape[1], shape[2], shape[3]), fill_value=fill_value, dtype="float32"
            )
        else:
            res = np.zeros((shape[0], shape[1], shape[2], shape[3]), dtype="float32")
        return torch.tensor(res, dtype=torch.float32)


def fft_of_levelsets(A, Q, packing, setup_fn):
    """
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """

    fft_list = []
    for a_start in range(0, Q, packing):
        a_end = np.minimum(a_start + packing, Q)
        levelsets = []
        for a in range(a_start, a_end):
            levelsets.append(float_compare(A, a))
        A_cat = torch.cat(levelsets, 0)
        del levelsets
        ffts = setup_fn(A_cat)
        del A_cat
        fft_list.append((ffts, a_start, a_end))
    return fft_list


def fft(A):
    """
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """

    spectrum = torch.fft.rfft2(A)
    return spectrum


def fftconv(A, B):
    """
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """

    C = A * B
    return C


def float_compare(A, c):
    """
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """
    return torch.clamp(1 - torch.abs(A - c), 0.0)


def ifft(A_fft):
    """
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """
    res = torch.fft.irfft2(A_fft)
    return res


def tf_rotate(I, angle, fill_value, center=None):
    """
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """
    if center is not None:
        center = [
            x + 0.5 for x in center
        ]  # Half a pixel offset, since TF.rotate origin is in upper left corner
    return TF.rotate(
        I,
        -angle,
        center=center,
        fill=[
            fill_value,
        ],
    )


def to_tensor(A, on_gpu=True):
    """
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """
    if torch.is_tensor(A):
        A_tensor = A.cuda(non_blocking=True) if on_gpu else A
        if A_tensor.ndim == 2:
            A_tensor = torch.reshape(A_tensor, (1, 1, A_tensor.shape[0], A_tensor.shape[1]))
        elif A_tensor.ndim == 3:
            A_tensor = torch.reshape(
                A_tensor, (1, A_tensor.shape[0], A_tensor.shape[1], A_tensor.shape[2])
            )
        return A_tensor
    else:
        return to_tensor(torch.tensor(A, dtype=VALUE_TYPE), on_gpu=on_gpu)
