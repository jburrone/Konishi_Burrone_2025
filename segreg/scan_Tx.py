#
# multimodal image registration
#

import argparse

import globalign
import numpy as np
from skimage import filters, io

from utils import globalign, register, segment, transform3d


def parse_args():
    parser = argparse.ArgumentParser(
        prog="scan_Tx.py",
        description="align modal-1 3d image to modal-2 2d/3d image.",
        add_help=True,
    )
    parser.add_argument("--ref-imgfile", type=str, help="file name of moving image")
    parser.add_argument("--ref-segfile", type=str, help="file name of moving segmentation")
    parser.add_argument("--mov-imgfile", type=str, help="file name of reference image")
    parser.add_argument("--mov-segfile", type=str, help="file name of reference segmentation")

    parser.add_argument("--x-offmov", default=350, type=int, help="bestfit x_offmov")
    parser.add_argument("--y-offmov", default=1150, type=int, help="bestfit y_offmov")

    parser.add_argument("--refine", default=0, type=int, help="whether refined (1) or not (0).")

    parser.add_argument("--tz-segreg", default=-0.7, type=float, help="bestfit tz_segreg")
    parser.add_argument("--ty-segreg", default=0.5, type=float, help="bestfit ty_segreg")

    parser.add_argument("--x-segreg", default=33, type=int, help="bestfit x_segreg")
    parser.add_argument("--y-segreg", default=-12, type=int, help="bestfit y_segreg")

    parser.add_argument("--z-segreg", default=8, type=int, help="bestfit z position")
    parser.add_argument("--sx-segreg", default=1.2, type=float, help="bestfit sx_segreg")
    parser.add_argument("--sy-segreg", default=1.05, type=float, help="bestfit sy_segreg")

    parser.add_argument(
        "--ref-scale-var",
        default=0.1,
        type=float,
        help="scale peak-to-valley value of moving image",
    )
    parser.add_argument("--angle", default=0, type=int, help="the center value of angle X search")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    iz_segreg = args.z_segreg
    ix_offmov, iy_offmov = args.x_offmov, args.y_offmov
    ix_segreg, iy_segreg = args.x_segreg, args.y_segreg
    sx, sy = args.sx_segreg, args.sy_segreg
    itz_segreg = args.tz_segreg
    ity_segreg = args.ty_segreg
    dz = 7

    ref_img = io.imread(args.ref_imgfile)
    if len(ref_img.shape) == 3:
        ref_img = ref_img[ref_img.shape[0] // 2]
    ref_seg = io.imread(args.ref_segfile)
    ref_seg = ((ref_seg > 0) * 255).astype(np.uint8)

    mov_img = io.imread(args.mov_imgfile)
    mov_seg = io.imread(args.mov_segfile)
    if mov_img.ndim == 2:
        mov_img = mov_img[np.newaxis, :, :]
    if mov_seg.ndim == 2:
        mov_seg = mov_seg[np.newaxis, :, :]
    mov_img_and_seg = mov_img, mov_seg

    mov_img_crop2d_Tx = []
    mov_seg_crop2d_Tx = []

    sxlist, sylist = [args.sx_segreg], [args.sy_segreg]
    sx_range = (args.sx_segreg - args.ref_scale_var, args.sx_segreg + args.ref_scale_var)
    sy_range = (args.sy_segreg - args.ref_scale_var, args.sy_segreg + args.ref_scale_var)
    if args.ref_scale_var != 0:
        sxlist, sylist = transform3d.generate_randomtf(
            n_transforms=2, sx_range=sx_range, sy_range=sy_range
        )

    augmented_ref_imgs = transform3d.apply_randomtf(ref_img, sxlist, sylist)
    augmented_ref_masks = transform3d.apply_randomtf(ref_seg, sxlist, sylist)

    for i in range(len(sxlist)):
        ref_img = transform3d.pad_image(augmented_ref_imgs[i])
        ref_seg = transform3d.pad_image(augmented_ref_masks[i])
        if ref_seg.ndim == 3:
            ref_img = ref_img[ref_img.shape[0] // 2]
            ref_seg = ref_seg[ref_seg.shape[0] // 2]
        ref_labels = segment.seg2label(ref_seg)
        ref_img_and_labels = ref_img, ref_labels

        mov_img_crop3d = np.zeros((2 * dz + 1,) + ref_img.shape, dtype=np.uint8)
        mov_seg_crop3d = np.zeros((2 * dz + 1,) + ref_labels.shape, dtype=np.uint8)
        if (
            iy_offmov + ref_labels.shape[1] < mov_seg.shape[1]
            and ix_offmov + ref_labels.shape[0] < mov_seg.shape[-1]
        ):

            mov_img_crop3d = mov_img[
                iz_segreg - dz : iz_segreg + dz + 1,
                iy_offmov : iy_offmov + ref_labels.shape[1],
                ix_offmov : ix_offmov + ref_labels.shape[0],
            ]
            mov_seg_crop3d = mov_seg[
                iz_segreg - dz : iz_segreg + dz + 1,
                iy_offmov : iy_offmov + ref_labels.shape[1],
                ix_offmov : ix_offmov + ref_labels.shape[0],
            ]

        param = [[0, itz_segreg, iy_segreg, ix_segreg, 0, 0]]

        mov_img_crop3d_reg = np.zeros((2 * dz + 1,) + ref_img.shape, dtype=np.uint8)
        mov_seg_crop3d_reg = np.zeros((2 * dz + 1,) + ref_labels.shape, dtype=np.uint8)
        for iz in range(mov_img_crop3d.shape[0]):
            mov_img_crop3d_reg[iz] = globalign.warp_image_rigid(
                ref_labels, mov_img_crop3d[iz], param[0], mode="spline", bg_value=0
            )
            mov_seg_crop3d_reg[iz] = globalign.warp_image_rigid(
                ref_labels, mov_seg_crop3d[iz], param[0], mode="spline", bg_value=0
            )

        mov_img_crop3d_reg_Ty = transform3d.rotate_image_center(
            mov_img_crop3d_reg, ity_segreg, axis="y"
        )
        mov_seg_crop3d_reg_Ty = transform3d.rotate_image_center(
            mov_seg_crop3d_reg, ity_segreg, axis="y"
        )
        mov_seg_crop3d_reg_Ty = (
            (mov_seg_crop3d_reg_Ty > filters.threshold_otsu(mov_seg_crop3d_reg_Ty)) * 255
        ).astype(np.uint8)

        mov_img_and_seg = mov_img_crop3d_reg_Ty, mov_seg_crop3d_reg_Ty
        register.scanTx(
            ref_img_and_labels, mov_img_and_seg, (sxlist[i], sylist[i]), args.angle, args.refine
        )
