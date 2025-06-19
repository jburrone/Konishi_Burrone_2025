#
# multimodal image registration x,y,z,Tz
#

import argparse

import numpy as np
from skimage import io

from utils import register, segment, transform3d


def parse_args():
    parser = argparse.ArgumentParser(
        prog="scan_xyzTz.py",
        description="scans 2/3d image against 2d image using registration.",
        add_help=True,
    )
    parser.add_argument("--mov-imgfile", type=str, help="Name of the reference image file. ")
    parser.add_argument(
        "--mov-segfile",
        type=str,
        help="Name of the reference segmentation file. ",
    )
    parser.add_argument("--ref-imgfile", type=str, help="Name of the moving image file. ")
    parser.add_argument(
        "--ref-segfile",
        type=str,
        help="Name of the moving segmentation file. ",
    )
    parser.add_argument(
        "--ref-scale-var",
        default=0.1,
        type=float,
        help="Scaling value limit of the reference image.",
    )

    parser.add_argument(
        "--xmi", default=0, type=int, help="Left bound pixel value of x search range."
    )
    parser.add_argument(
        "--xma",
        default=1024,
        type=int,
        help="Right bound pixel value of x search range.",
    )
    parser.add_argument(
        "--ymi", default=0, type=int, help="Top bound pixel value of y search range."
    )
    parser.add_argument(
        "--yma",
        default=1024,
        type=int,
        help="Bottom bound pixel value of y search range.",
    )
    parser.add_argument(
        "--zmi", default=16, type=int, help="Minimum z pxiel value of z search range."
    )
    parser.add_argument(
        "--zma", default=17, type=int, help="Maximum z pxiel value of z search range."
    )
    parser.add_argument("--zstep", default=1, type=int, help="Step size for the z search.")
    parser.add_argument(
        "--angle",
        default=1,
        type=int,
        help="Center value of the z angle search in degrees.",
    )

    parser.add_argument(
        "--refine",
        default=0,
        type=int,
        help="whether image registration is refined (1) or not (0).",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Search volume
    registration_region = (
        args.xmi,
        args.xma,
        args.ymi,
        args.yma,
        args.zmi,
        args.zma,
        args.zstep,
    )

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

    sx_segreg, sy_segreg = 1, 1
    sxlist, sylist = [sx_segreg], [sy_segreg]
    # TODO:
    # sx_range = (sx_segreg - args.ref_scale_var, sx_segreg + args.ref_scale_var)
    # sy_range = (sy_segreg - args.ref_scale_var, sy_segreg + args.ref_scale_var)
    sx_range = (sx_segreg, sx_segreg + args.ref_scale_var)
    sy_range = (sy_segreg, sy_segreg + args.ref_scale_var)

    if args.ref_scale_var != 0:
        sxlist, sylist = transform3d.generate_randomtf(
            n_transforms=10, sx_range=sx_range, sy_range=sy_range
        )

    augmented_ref_imgs = transform3d.apply_randomtf(ref_img, sxlist, sylist)
    augmented_ref_masks = transform3d.apply_randomtf(ref_seg, sxlist, sylist)
    for i in range(len(sxlist)):
        ref_img = transform3d.pad_image(augmented_ref_imgs[i])
        ref_seg = transform3d.pad_image(augmented_ref_masks[i])
        if ref_seg.ndim == 3:
            ref_seg = ref_seg[ref_seg.shape[0] // 2]

        ref_labels = segment.seg2label(ref_seg)

        ref_img_and_labels = ref_img, ref_labels

        register.scanxyzTz(
            ref_img_and_labels,
            mov_img_and_seg,
            (sxlist[i], sylist[i]),
            registration_region,
            args.angle,
            args.refine,
        )
