import numpy as np
import torch
import torch.fft
import torch.nn.functional as F

from utils import transform2d, util_gpu


def grid_angles(angle_center, angle_dev, n=32):
    """
    Creates a list of angles.
    Args:
        angle_center (float): the center value of angle range in degree.
        angle_dev (float): the deviation value from its center in dgree.
        n (int, optional): number of generated grid angle. Defaults to 32.
    Returns:
        angles (list): list of angles.
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """
    angles = []
    n_denom = n
    if angle_dev < 180:
        n_denom -= 1
    for i in range(n):
        i_frac = i / n_denom
        angle = angle_center + (2.0 * i_frac - 1.0) * angle_dev
        angles.append(angle)
    return angles


def random_angles(centers, center_prob, angle_dev, n=32):
    """
    Genreates list of random angles.
    Args:
        centers (list): list of center angles.
        center_prob (list): list of probabilities for each center angles.
        angle_dev (float): the deviation value from its center in dgree.
        n (int, optional): number of generated grid angle. Defaults to 32.
    Returns:
        angles (list): list of random angles.
    Original:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    Modification:
        rng -> np.random
    """
    angles = []
    if not isinstance(centers, list):
        centers = [centers]
    if center_prob is not None:
        mass = np.sum(center_prob)
        p = center_prob / mass
    else:
        p = None
    for _ in range(n):
        c = np.random.choice(centers, p=p, replace=True)
        frac = np.random.random()
        angle = c + (2.0 * frac - 1.0) * angle_dev
        angles.append(angle)
    return angles


def align_rigid(
    A,
    B,
    M_A,
    M_B,
    Q_A,
    Q_B,
    angles,
    overlap=0.5,
    enable_partial_overlap=True,
    normalize_mi=False,
    on_gpu=True,
    save_maps=False,
):
    """
    Performs rigid alignment of multimodal images
    using exhaustive search mutual information (MI),
    locating the global maximum of the MI measure
    w.r.t. all possible whole-pixel translations as well
    as a set of enumerated rotations.
    Runs on the GPU, using PyTorch.
    Args:
        A (array): reference 2d/3d image.
        B (array): moving 2d/3d image.
        M_A (array): reference 2d mask, the user-defined part to be included in the computation of MI.
        M_B (array): moving 2d mask, the user-defined part to be included in the computation of MI.
        Q_A (int): number of quantization levels in image A.
        Q_B (int): number of quantization levels in image B.
        angles (list): List of angles for the rigid alignment.
        overlap (float, optional): The required overlap fraction.
        enable_partial_overlap (bool, optional):
            If False then no padding will be done,
            and only fully overlapping configurations will be evaluated.
            If True, then padding will be done to include configurations
            where only part of image B is overlapping image A.
        normalize_mi (bool, optional): Flag to choose between normalized MI
            or standard unnormalized mutual information.
        on_gpu (bool, optional): Flag controlling
            if the alignment is done on the GPU.
        save_maps (bool, optional): Flag for exporting the stack of
            CMIF maps over the angles, e.g. for debugging or visualization.
    Returns:
        list of tuples: tuple includes
            (np.array(mutial information), angle, offset y, offset x, cy, cx)
        maps/None.
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
        Normalized MI: Knops et al. 2006, Medical Image Analysis
        https://www.sciencedirect.com/science/article/pii/S1361841505000435
    """

    eps = 1e-7
    results = []
    maps = []
    A_tensor = util_gpu.to_tensor(A, on_gpu=on_gpu)
    B_tensor = util_gpu.to_tensor(B, on_gpu=on_gpu)

    packing = np.minimum(Q_B, 64)

    # Create all constant masks if not provided
    if M_A is None:
        M_A = util_gpu.create_float_tensor(A_tensor.shape, on_gpu, 1.0)
    else:
        M_A = util_gpu.to_tensor(M_A, on_gpu)
        A_tensor = torch.round(M_A * A_tensor + (1 - M_A) * (Q_A + 1))
    if M_B is None:
        M_B = util_gpu.create_float_tensor(B_tensor.shape, on_gpu, 1.0)
    else:
        M_B = util_gpu.to_tensor(M_B, on_gpu)

    # Pad for overlap
    if enable_partial_overlap:
        partial_overlap_pad_size = (
            round(B.shape[-1] * (1.0 - overlap)),  # padsize
            round(B.shape[-2] * (1.0 - overlap)),  # padsize
        )
        A_tensor = F.pad(
            A_tensor,
            (
                partial_overlap_pad_size[0],  # padding_left
                partial_overlap_pad_size[0],  # padding_right
                partial_overlap_pad_size[1],  # padding_top
                partial_overlap_pad_size[1],  # padding_bottom
            ),
            mode="constant",
            value=Q_A + 1,
        )
        M_A = F.pad(
            M_A,
            (
                partial_overlap_pad_size[0],  # padding_left
                partial_overlap_pad_size[0],  # padding_right
                partial_overlap_pad_size[1],  # padding_top
                partial_overlap_pad_size[1],  # padding_bottom
            ),
            mode="constant",
            value=0,
        )
    else:
        partial_overlap_pad_size = (0, 0)

    A_shape = A_tensor.shape  # [1, 1, sz + 2*padsize, sz + 2*padsize]
    B_shape = B_tensor.shape  # [1, 1, sz, sz]
    Bpad_shape = torch.tensor(A_shape, dtype=torch.long) - torch.tensor(
        B_shape, dtype=torch.long
    )  # [1, 1, 2*padsize, 2*padsize]
    ext_valid_shape = Bpad_shape + 1  # [1, 1, 2*padsize + 1, 2*padsize + 1]
    batched_valid_shape = ext_valid_shape + torch.tensor(
        [packing - 1, 0, 0, 0]
    )  # [packing, 1, 2*padsize + 1, 2*padsize + 1]

    center_of_rotation = [B_shape[3] / 2.0, B_shape[2] / 2.0]

    M_A_fft = util_gpu.corr_target_setup(M_A)  # [1, 1, sz, sz/2]

    A_ffts = []  # [1, 1, sz, sz/2]
    for a in range(Q_A):
        A_ffts.append(util_gpu.corr_target_setup(util_gpu.float_compare(A_tensor, a)))
    del A_tensor
    del M_A

    if normalize_mi:
        H_MARG = util_gpu.create_float_tensor(ext_valid_shape, on_gpu, 0.0)
        H_AB = util_gpu.create_float_tensor(ext_valid_shape, on_gpu, 0.0)
    else:
        MI = util_gpu.create_float_tensor(ext_valid_shape, on_gpu, 0.0)

    for angle in angles:
        B_tensor_rotated = util_gpu.tf_rotate(B_tensor, angle, Q_B, center=center_of_rotation)
        M_B_rotated = util_gpu.tf_rotate(M_B, angle, 0, center=center_of_rotation)
        B_tensor_rotated = torch.round(
            M_B_rotated * B_tensor_rotated + (1 - M_B_rotated) * (Q_B + 1)
        )
        B_tensor_rotated = F.pad(  # [1, 1, sz + 2*padsize, sz + 2*padsize]
            B_tensor_rotated,
            (
                0,  # padding_left
                A_shape[-1] - B_shape[-1],  # padding_right
                0,  # padding_top
                A_shape[-2] - B_shape[-2],  # padding_bottom
                0,  # padding_front
                0,  # padding_back
                0,
                0,
            ),
            mode="constant",
            value=Q_B + 1,
        )
        M_B_rotated = F.pad(  # [1, 1, sz + 2*padsize, sz + 2*padsize]
            M_B_rotated,
            (
                0,  # padding_left
                A_shape[-1] - B_shape[-1],  # padding_right
                0,  # padding_top
                A_shape[-2] - B_shape[-2],  # padding_bottom
                0,  # padding_front
                0,  # padding_back
                0,
                0,
            ),
            mode="constant",
            value=0,
        )

        M_B_fft = util_gpu.corr_template_setup(M_B_rotated)  # [1, 1, sz, sz/2]
        del M_B_rotated

        # Eq. (11) of Oefverstedt et al. 2022
        N = torch.clamp(
            util_gpu.corr_apply(M_A_fft, M_B_fft, ext_valid_shape), min=eps, max=None
        )  # [1, 1, 2*padsize + 1, 2*padsize + 1]

        B_ffts = util_gpu.fft_of_levelsets(  # [Q_B, 1, sz, sz/2]
            B_tensor_rotated, Q_B, packing, util_gpu.corr_template_setup
        )

        for bext in range(len(B_ffts)):
            # C_bext/N * log2(C_bext/N) in Eq. (12) of Oefverstedt et al. 2022
            B_fft = B_ffts[bext]
            E_M = torch.sum(  # [1, 1, 2*padsize + 1, 2*padsize + 1]
                util_gpu.compute_entropy(  # [packing, 1, 2*padsize + 1, 2*padsize + 1]
                    # C_bext in Eq. (10) of Oefverstedt et al. 2022
                    util_gpu.corr_apply(M_A_fft, B_fft[0], batched_valid_shape),
                    N,
                    eps,
                ),
                dim=0,
            )
            if normalize_mi:
                H_MARG = torch.sub(H_MARG, E_M)
            else:
                MI = torch.sub(MI, E_M)
            del E_M

            for a in range(Q_A):
                # C_a in Eq. (10) of Oefverstedt et al. 2022
                # C_a/N * log2(C_a/N) in Eq. (12) of Oefverstedt et al. 2022
                A_fft_cuda = A_ffts[a]
                if bext == 0:
                    E_M = util_gpu.compute_entropy(  # [1, 1, 2*padsize + 1, 2*padsize + 1]
                        util_gpu.corr_apply(A_fft_cuda, M_B_fft, ext_valid_shape),
                        N,
                        eps,
                    )
                    if normalize_mi:
                        H_MARG = torch.sub(H_MARG, E_M)
                    else:
                        MI = torch.sub(MI, E_M)
                    del E_M

                # C_{a,bext}/N * log2(C_{a,bext}/N) in Eq. (12) of Oefverstedt et al. 2022
                E_J = torch.sum(  # [1, 2*padsize + 1, 2*padsize + 1]
                    util_gpu.compute_entropy(  # [packing, 1, 2*padsize + 1, 2*padsize + 1]
                        # C_{a,b} in Eq. (9) of Oefverstedt et al. 2022
                        util_gpu.corr_apply(A_fft_cuda, B_fft[0], batched_valid_shape),
                        N,
                        eps,
                    ),
                    dim=0,
                )

                if normalize_mi:
                    H_AB = torch.sub(H_AB, E_J)
                else:
                    # MI = H_A + H_B - H_AB in Eq. (2) of Oefverstedt et al. 2022
                    MI = torch.add(MI, E_J)
                del E_J
                del A_fft_cuda
            del B_fft
            if bext == 0:
                del M_B_fft

        del B_tensor_rotated

        if normalize_mi:
            # normalized MI = MI / H_AB
            # H_MARG / H_AB: Only one element tensors can be converted to scalar
            MI = torch.clamp((H_MARG / (H_AB + eps) - 1), 0.0, 1.0)

        if save_maps:
            maps.append(MI.cpu().numpy())

        (max_n, _) = torch.max(torch.reshape(N, (-1,)), 0)
        N_filt = torch.lt(N, overlap * max_n)
        MI[N_filt] = 0.0
        del N_filt, N

        MI_vec = torch.reshape(MI, (-1,))
        # _mi_vec = MI_vec.to('cpu').detach().numpy().copy()

        (val, ind) = torch.max(MI_vec, -1)

        results.append((angle, val, ind))

        if normalize_mi:
            H_MARG.fill_(0)
            H_AB.fill_(0)
        else:
            MI.fill_(0)

    results_cpu = []
    for i in range(len(results)):
        angle = results[i][0]
        maxval = results[i][1].cpu().numpy()
        maxind = results[i][2].cpu().numpy()
        sz_x = int(ext_valid_shape[3].numpy())
        y = maxind // sz_x
        x = maxind % sz_x
        results_cpu.append(
            (
                maxval,
                angle,
                -(y - partial_overlap_pad_size[1]),
                -(x - partial_overlap_pad_size[0]),
                center_of_rotation[1],
                center_of_rotation[0],
            )
        )
    results_cpu = sorted(results_cpu, key=(lambda tup: tup[0]), reverse=True)

    if save_maps:
        return results_cpu, maps
    else:
        return results_cpu, None


def warp_image_rigid(
    ref_image, flo_image, param, mode="nearest", bg_value=0.0, inverse_transform=False
):
    """
    Applies the transformation obtained by the functions align_rigid
    to warp a moving image into the space of the ref_image (using backward mapping).

    Args:
        ref_image (array): reference 2d image.
        flo_image (array): moving 2d image.
        param (tuple): The returned tuple from function: align_rigid.
        mode (str, optional): interpolation mode, nearest/linear/spline.
            Defaults to 'nearest'.
        bg_value (float, optional): The value to insert where
            there is no information in the flo_image. Defaults to 0.0.
        inverse_transform (bool, optional): Invert the transformation, used
            e.g. when warping the original reference image
            into the space of the original moving image. Defaults to False.
    Returns:
        flo_image_out (array): registered moving 2d image,
            If angle(param[1]) > 0, the bright point moves clockwise around a point
            in the image (param[4], param[5]);
            If param[2] > 0, it moves upwards;
            If param[3] > 0, it moves leftwards.
    Citation:
        https://github.com/MIDA-group/globalign/blob/main/globalign.py
    """

    r = transform2d.Rotate2DTransform()
    r.set_param(0, np.pi * param[1] / 180.0)
    translation = transform2d.TranslationTransform(2)
    translation.set_param(0, param[2])
    translation.set_param(1, param[3])
    t = transform2d.CompositeTransform(2, [translation, r])
    t = transform2d.make_centered_transform(t, np.array(param[4:]), np.array(param[4:]))

    if inverse_transform:
        t = t.invert()

    out_shape = ref_image.shape[:2] + flo_image.shape[2:]
    flo_image_out = np.zeros(out_shape, dtype=flo_image.dtype)

    if flo_image.ndim == 3:
        for i in range(flo_image.shape[2]):
            bg_val_i = np.array(bg_value)
            if bg_val_i.shape[0] == flo_image.shape[2]:
                bg_val_i = bg_val_i[i]
            t.warp(
                flo_image[:, :, i],
                flo_image_out[:, :, i],
                in_spacing=np.ones(
                    2,
                ),
                out_spacing=np.ones(
                    2,
                ),
                mode=mode,
                bg_value=bg_val_i,
            )
    else:
        t.warp(
            flo_image,
            flo_image_out,
            in_spacing=np.ones(
                2,
            ),
            out_spacing=np.ones(
                2,
            ),
            mode=mode,
            bg_value=bg_value,
        )

    return flo_image_out
