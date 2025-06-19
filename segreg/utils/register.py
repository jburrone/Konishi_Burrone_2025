import csv
from datetime import datetime

from utils import globalign, segment, transform3d, util_cpu
from skimage import filters, io
import numpy as np

# Define the header as a constant list
HEADER_scanxyzTz = [
    "z_segreg",
    "y_offmov",
    "y_segreg",
    "x_offmov",
    "x_segreg",
    "tz_segreg",
    "sx_segreg",
    "sy_segreg",
    "mi",
    "opixmatch",
    "jaccard",
    "omatch",
    "ce",
    "all",
]

HEADER_scanTy = [
    "ty_segreg",
    "y_segreg",
    "x_segreg",
    "sx_segreg",
    "sy_segreg",
    "mi",
    "opixmatch",
    "jaccard",
    "omatch",
    "ce",
    "all",
]

HEADER_scanTx = [
    "tx_segreg",
    "y_segreg",
    "x_segreg",
    "sx_segreg",
    "sy_segreg",
    "mi",
    "opixmatch",
    "jaccard",
    "omatch",
    "ce",
    "all",
]


def register(ref_img, mov_img, overlap, angle_center, angle_stdev, k=8, n=3):
    """
    Registers multimodal image using CMIF
    Args:
        ref_img (array): reference image, gray
        mov_img (array): moving image, gray
        overlap (float): allowed overlap between ref_img and mov_img
        angle_center (int, optional): the center value of angle search
        angle_stdev (int): allowed angle range, in deg
        k (int, optional): number of clustering. Defaults to 8.
        n (int, optional): number of registered image candidates
    Returns:
        array: registered flo image
        param (tuple): The returned tuple from function: align_rigid.
    """
    sz = ref_img.shape[0]

    quantized_ref_img = ref_img.astype(np.int32)
    quantized_mov_img = mov_img.astype(np.int32)

    M_ref = np.ones((sz, sz), dtype="bool")
    M_flo = np.ones((sz, sz), dtype="bool")

    angle_candidates = globalign.grid_angles(angle_center, angle_stdev, 100)

    param, _ = globalign.align_rigid(
        quantized_ref_img,
        quantized_mov_img,
        M_ref,
        M_flo,
        k,
        k,
        angle_candidates,
        overlap=overlap,
        enable_partial_overlap=True,
        normalize_mi=True,
        on_gpu=True,
        save_maps=True,
    )

    mov_img_regs = []
    for i in range(n):
        mov_img_reg = globalign.warp_image_rigid(
            ref_img, mov_img, param[i], mode="nearest", bg_value=1
        )
        mov_img_regs.append(mov_img_reg)
    return mov_img_regs, param


def scanxyzTz(
    ref_img_and_labels,
    mov_img_and_seg,
    sxy_segreg,
    registration_region,
    angle_center,
    flag_refine=0,
    flag_save=1,
):
    """
    Searches for the bestmatch in the z direction
    Args:
        ref_img_and_labels (list of array): reference image and label.
        mov_img_and_seg (list of array): moving image and seg.
        sxy_segreg (tuple): scale x and scale y
        registration_region:
            xmi (int): the coordinate of left location for scan
            xma (int): the coordinate of right location for scan
            ymi (int): the coordinate of top location for scan
            yma (int): the coordinate of bottom location for scan
            zmi (int): the first page of the image stack
            zma (int): the last page of the image stack
            dz (int): zstep of the image stack
        angle_center (int): the center value of angle search
        flag_refine (int, optional): whether refined (1) or not (0).
        flag_save (int, optional): Whether the results saved (1) or not (0).
    Returns:
        _type_: _description_
    """

    xmi, xma, ymi, yma, zmi, zma, dz = registration_region
    z_search_range = range(zmi, zma, dz)

    ref_img, ref_labels = ref_img_and_labels
    mov_img, mov_seg = mov_img_and_seg
    isx_segreg, isy_segreg = sxy_segreg

    now = datetime.now()
    ymdhms = now.strftime("%Y%m%d%H%M%S")
    logfilename = f"align_xyzTz_{ymdhms}.csv"

    f = open(logfilename, "a")
    writer = csv.writer(f)
    writer.writerow(HEADER_scanxyzTz)

    ofsty, ofstx = int(ref_labels.shape[0] // 8), int(ref_labels.shape[1] // 8)
    if ymi < yma - ref_labels.shape[0] + 1:
        iylist = range(ymi, yma - ref_labels.shape[0] + 1, ofsty)
    else:
        iylist = [ymi]
    if xmi < xma - ref_labels.shape[1] + 1:
        ixlist = range(xmi, xma - ref_labels.shape[1] + 1, ofstx)
    else:
        ixlist = [xmi]

    total_scan = len(iylist) * len(ixlist) * len(z_search_range)
    print(f"Total scan number: {total_scan}")

    if flag_refine == 1:
        overlap, angle_stdev = 0.8, 3
    else:
        overlap, angle_stdev = 0.8, 3

    for iz in z_search_range:
        if iz % 5 == 1:
            print(f"Trying z: {iz}/{zma}")

        for _, pair in util_cpu.enumerated_product(iylist, ixlist):
            iy_offmov, ix_offmov = pair

            mov_seg_crop2d = np.zeros(ref_labels.shape)
            if (
                iy_offmov + ref_labels.shape[1] < mov_seg.shape[1]
                and ix_offmov + ref_labels.shape[0] < mov_seg.shape[-1]
            ):
                mov_seg_crop2d = mov_seg[
                    iz,
                    iy_offmov : iy_offmov + ref_labels.shape[1],
                    ix_offmov : ix_offmov + ref_labels.shape[0],
                ]

            ix_segreg = 0
            iy_segreg = 0
            jaccard = 0
            loss_ce = 0
            itz_segreg = 0
            objpix_match_ratio = 0
            obj_match_ratio = 0
            param = [[0, 0, 0, 0, 0, 0]]
            loss_all = 0

            mov_seg_crop2d_maxvalue = np.max(mov_seg_crop2d.flatten())
            if mov_seg_crop2d_maxvalue > 0:
                mov_labels = segment.seg2label(mov_seg_crop2d)
                k1 = max(np.max(ref_labels), np.max(mov_labels)) + 1
                _, param = register(ref_labels, mov_labels, overlap, angle_center, angle_stdev, k1)

                itz_segreg, iy_segreg, ix_segreg = param[0][1], param[0][2], param[0][3]

                mov_labels_reg = globalign.warp_image_rigid(
                    ref_labels, mov_labels, param[0], mode="nearest", bg_value=1.0
                )

                objpix_match_ratio, jaccard, obj_match_ratio = segment.calculate_metrics(
                    (ref_labels > 1).astype(np.uint8) * 255,
                    (mov_labels_reg > 1).astype(np.uint8) * 255,
                )
                loss_objpixmatch = 1 - objpix_match_ratio
                loss_objmatch = 1 - obj_match_ratio

                mov_img_crop2d = mov_img[
                    iz,
                    iy_offmov : iy_offmov + ref_labels.shape[1],
                    ix_offmov : ix_offmov + ref_labels.shape[0],
                ]
                mov_img_crop2d_reg = globalign.warp_image_rigid(
                    ref_labels, mov_img_crop2d, param[0], mode="nearest", bg_value=1.0
                )
                loss_ce = segment.cross_entropy(
                    ref_labels,
                    mov_labels_reg,
                    ref_img,
                    mov_img_crop2d_reg,
                    weight_bg_to_obj=0.1,
                )

                mov_seg_crop2d = mov_seg[
                    iz,
                    iy_offmov : iy_offmov + ref_labels.shape[1],
                    ix_offmov : ix_offmov + ref_labels.shape[0],
                ]
                mov_seg_crop2d = segment.relabel(mov_seg_crop2d)
                mov_seg_crop2d_reg = globalign.warp_image_rigid(
                    ref_labels, mov_seg_crop2d, param[0], mode="nearest", bg_value=1.0
                )

                overlap12_img = segment.overlap_images(ref_img, mov_img_crop2d_reg)
                overlap12_seg = segment.overlap_images(ref_labels, mov_seg_crop2d_reg)

                loss_all = loss_objpixmatch + loss_objmatch + loss_ce

                if flag_save == 1:
                    filename_template = f"overlap_reg_z{iz:02d}_y{iy_offmov:03d}_x{ix_offmov:03d}_tz{itz_segreg:04.1f}_mi{param[0][0]:3.3f}_opixmatch{objpix_match_ratio:0.2f}_jaccard{jaccard:.2f}_omatch{obj_match_ratio:0.2f}_ce{loss_ce:.2f}"
                    filename_img = f"{filename_template}.tif"
                    filename_seg = f"{filename_template}_seg.tif"

                    io.imsave(filename_img, overlap12_img, check_contrast=False)
                    io.imsave(filename_seg, overlap12_seg, check_contrast=False)

            writer.writerow(
                [
                    iz,
                    iy_offmov,
                    iy_segreg,
                    ix_offmov,
                    ix_segreg,
                    itz_segreg,
                    isx_segreg,
                    isy_segreg,
                    param[0][0],
                    objpix_match_ratio,
                    jaccard,
                    obj_match_ratio,
                    loss_ce,
                    loss_all,
                ]
            )
    return


def scanTy(
    ref_img_and_labels,
    mov_img_and_seg,
    sxy_segreg,
    angle_center,
    flag_refine=0,
    flag_save=1,
):
    """
    Searches for the bestmatch in the Ty direction
    Args:
        ref_img_and_labels (list of array): reference image and label.
        mov_img_and_seg (list of array): moving image and seg.
        sxy_segreg (tuple): scale x and scale y
        angle_center (int): the center value of angle search
        flag_refine (int, optional): whether refined (1) or not (0).
        flag_save (int, optional): Whether the results saved (1) or not (0).
    Returns:
        _type_: _description_
    """

    ref_img, ref_labels = ref_img_and_labels
    mov_img_crop3d_reg, mov_seg_crop3d_reg = mov_img_and_seg
    isx_segreg, isy_segreg = sxy_segreg

    now = datetime.now()
    ymdhms = now.strftime("%Y%m%d%H%M%S")
    logfilename = f"align_Ty_{ymdhms}.csv"

    f = open(logfilename, "a")
    writer = csv.writer(f)
    writer.writerow(HEADER_scanTy)

    angle_candidates = globalign.grid_angles(angle_center=angle_center, angle_dev=1.5, n=5)

    for ity_segreg in angle_candidates:
        mov_img_crop3d_reg_Ty = transform3d.rotate_image_center(
            mov_img_crop3d_reg, ity_segreg, axis="y"
        )
        mov_seg_crop3d_reg_Ty = transform3d.rotate_image_center(
            mov_seg_crop3d_reg, ity_segreg, axis="y"
        )
        mov_seg_crop3d_reg_Ty = (
            (mov_seg_crop3d_reg_Ty > filters.threshold_otsu(mov_seg_crop3d_reg_Ty)) * 255
        ).astype(np.uint8)

        mov_img_crop2d_Ty = mov_img_crop3d_reg_Ty[mov_img_crop3d_reg_Ty.shape[0] // 2]
        mov_seg_crop2d_Ty = mov_seg_crop3d_reg_Ty[mov_seg_crop3d_reg_Ty.shape[0] // 2]

        jaccard = 0
        loss_ce = 0
        objpix_match_ratio = 0
        obj_match_ratio = 0
        loss_all = 0

        mov_seg_crop2d_Ty_maxvalue = np.max(mov_seg_crop2d_Ty.flatten())
        if mov_seg_crop2d_Ty_maxvalue > 0:
            mov_labels = segment.seg2label(mov_seg_crop2d_Ty)
            k1 = max(np.max(ref_labels), np.max(mov_labels)) + 1
            # _, param = register(ref_labels, mov_labels, overlap=0.95, angle_center=0, angle_stdev=0, k1=k1)
            _, param = register(ref_labels, mov_labels, 0.8, 0, 0, k1)

            _, iy_segreg, ix_segreg = param[0][1], param[0][2], param[0][3]

            mov_labels_reg = globalign.warp_image_rigid(
                ref_labels, mov_labels, param[0], mode="spline", bg_value=1.0
            )

            objpix_match_ratio, jaccard, obj_match_ratio = segment.calculate_metrics(
                (ref_labels > 1).astype(np.uint8) * 255,
                (mov_labels_reg > 1).astype(np.uint8) * 255,
            )
            loss_objpixmatch = 1 - objpix_match_ratio
            loss_objmatch = 1 - obj_match_ratio

            mov_img_crop2d_reg = globalign.warp_image_rigid(
                ref_labels, mov_img_crop2d_Ty, param[0], mode="spline", bg_value=1.0
            )
            loss_ce = segment.cross_entropy(
                ref_labels,
                mov_labels_reg,
                ref_img,
                mov_img_crop2d_reg,
                weight_bg_to_obj=0.1,
            )

            mov_seg_crop2d = segment.relabel(mov_seg_crop2d_Ty)
            mov_seg_crop2d_reg = globalign.warp_image_rigid(
                ref_labels, mov_seg_crop2d, param[0], mode="spline", bg_value=1.0
            )

            overlap12_img = segment.overlap_images(ref_img, mov_img_crop2d_reg)
            overlap12_seg = segment.overlap_images(ref_labels, mov_seg_crop2d_reg)

            loss_all = loss_objpixmatch + loss_objmatch + loss_ce

            filename_template = f"overlap_reg_ty{ity_segreg:04.1f}_mi{param[0][0]:3.3f}_opixmatch{objpix_match_ratio:0.2f}_jaccard{jaccard:.2f}_omatch{obj_match_ratio:0.2f}_ce{loss_ce:.2f}"
            filename_img = f"{filename_template}.tif"
            filename_seg = f"{filename_template}_seg.tif"
            # filename_img = (
            #     filename_template
            #     % (
            #         ity_segreg,
            #         param[0][0],
            #         objpix_match_ratio,
            #         jaccard,
            #         obj_match_ratio,
            #         loss_ce,
            #     )
            #     + ".tif"
            # )

            # filename_seg = (
            #     filename_template
            #     % (
            #         ity_segreg,
            #         param[0][0],
            #         objpix_match_ratio,
            #         jaccard,
            #         obj_match_ratio,
            #         loss_ce,
            #     )
            #     + "_seg.tif"
            # )

            io.imsave(filename_img, overlap12_img, check_contrast=False)
            io.imsave(filename_seg, overlap12_seg, check_contrast=False)

        writer.writerow(
            [
                ity_segreg,
                iy_segreg,
                ix_segreg,
                isx_segreg,
                isy_segreg,
                param[0][0],
                objpix_match_ratio,
                jaccard,
                obj_match_ratio,
                loss_ce,
                loss_all,
            ]
        )


def scanTx(
    ref_img_and_labels,
    mov_img_and_seg,
    sxy_segreg,
    angle_center,
    flag_refine=0,
    flag_save=1,
):
    """
    Searches for the bestmatch in the Tx direction
    Args:
        ref_img_and_labels (list of array): reference image and label.
        mov_img_and_seg (list of array): moving image and seg.
        sxy_segreg (tuple): scale x and scale y
        angle_center (int): the center value of angle search
        flag_refine (int, optional): whether refined (1) or not (0).
        flag_save (int, optional): Whether the results saved (1) or not (0).
    Returns:
        _type_: _description_
    """

    ref_img, ref_labels = ref_img_and_labels
    mov_img_crop3d_reg, mov_seg_crop3d_reg = mov_img_and_seg
    isx_segreg, isy_segreg = sxy_segreg

    now = datetime.now()
    ymdhms = now.strftime("%Y%m%d%H%M%S")
    logfilename = f"align_Tx_{ymdhms}.csv"

    f = open(logfilename, "a")
    writer = csv.writer(f)
    writer.writerow(HEADER_scanTx)

    angle_candidates = globalign.grid_angles(angle_center=angle_center, angle_dev=1.5, n=5)

    for itx_segreg in angle_candidates:
        mov_img_crop3d_reg_Tx = transform3d.rotate_image_center(
            mov_img_crop3d_reg, itx_segreg, axis="x"
        )
        mov_seg_crop3d_reg_Tx = transform3d.otate_image_center(
            mov_seg_crop3d_reg, itx_segreg, axis="x"
        )
        mov_seg_crop3d_reg_Tx = (
            (mov_seg_crop3d_reg_Tx > filters.threshold_otsu(mov_seg_crop3d_reg_Tx)) * 255
        ).astype(np.uint8)

        mov_img_crop2d_Tx = mov_img_crop3d_reg_Tx[mov_img_crop3d_reg_Tx.shape[0] // 2]
        mov_seg_crop2d_Tx = mov_seg_crop3d_reg_Tx[mov_seg_crop3d_reg_Tx.shape[0] // 2]

        jaccard = 0
        loss_ce = 0
        objpix_match_ratio = 0
        obj_match_ratio = 0
        loss_all = 0

        mov_seg_crop2d_Tx_maxvalue = np.max(mov_seg_crop2d_Tx.flatten())
        if mov_seg_crop2d_Tx_maxvalue > 0:
            mov_labels = segment.seg2label(mov_seg_crop2d_Tx)
            k1 = max(np.max(ref_labels), np.max(mov_labels)) + 1
            # _, param = register(ref_labels, mov_labels, overlap=0.95, angle_center=0, angle_stdev=0, k1=k1)
            _, param = register(ref_labels, mov_labels, 0.8, 0, 0, k1)

            _, iy_segreg, ix_segreg = param[0][1], param[0][2], param[0][3]

            mov_labels_reg = globalign.warp_image_rigid(
                ref_labels, mov_labels, param[0], mode="spline", bg_value=1.0
            )

            objpix_match_ratio, jaccard, obj_match_ratio = segment.calculate_metrics(
                (ref_labels > 1).astype(np.uint8) * 255,
                (mov_labels_reg > 1).astype(np.uint8) * 255,
            )
            loss_objpixmatch = 1 - objpix_match_ratio
            loss_objmatch = 1 - obj_match_ratio

            mov_img_crop2d_reg = globalign.warp_image_rigid(
                ref_labels, mov_img_crop2d_Tx, param[0], mode="spline", bg_value=1.0
            )
            loss_ce = segment.cross_entropy(
                ref_labels,
                mov_labels_reg,
                ref_img,
                mov_img_crop2d_reg,
                weight_bg_to_obj=0.1,
            )

            mov_seg_crop2d = segment.relabel(mov_seg_crop2d_Tx)
            mov_seg_crop2d_reg = globalign.warp_image_rigid(
                ref_labels, mov_seg_crop2d, param[0], mode="spline", bg_value=1.0
            )

            overlap12_img = segment.overlap_images(ref_img, mov_img_crop2d_reg)
            overlap12_seg = segment.overlap_images(ref_labels, mov_seg_crop2d_reg)

            loss_all = loss_objpixmatch + loss_objmatch + loss_ce

            # filename_template = "overlap_reg_tx%04.1f_mi%3.3f_opixmatch%0.2f_jaccard%.2f_omatch%0.2f_ce%.2f"
            filename_template = f"overlap_reg_tx{itx_segreg:04.1f}_mi{param[0][0]:3.3f}_opixmatch{objpix_match_ratio:0.2f}_jaccard{jaccard:.2f}_omatch{obj_match_ratio:0.2f}_ce{loss_ce:.2f}"
            filename_img = f"{filename_template}.tif"
            filename_seg = f"{filename_template}_seg.tif"
            # filename_img = (
            #     filename_template
            #     % (
            #         itx_segreg,
            #         param[0][0],
            #         objpix_match_ratio,
            #         jaccard,
            #         obj_match_ratio,
            #         loss_ce,
            #     )
            #     + ".tif"
            # )

            # filename_seg = (
            #     filename_template
            #     % (
            #         itx_segreg,
            #         param[0][0],
            #         objpix_match_ratio,
            #         jaccard,
            #         obj_match_ratio,
            #         loss_ce,
            #     )
            #     + "_seg.tif"
            # )

            io.imsave(filename_img, overlap12_img, check_contrast=False)
            io.imsave(filename_seg, overlap12_seg, check_contrast=False)

        writer.writerow(
            [
                itx_segreg,
                iy_segreg,
                ix_segreg,
                isx_segreg,
                isy_segreg,
                param[0][0],
                objpix_match_ratio,
                jaccard,
                obj_match_ratio,
                loss_ce,
                loss_all,
            ]
        )
