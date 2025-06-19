#
# multimodal image registration x,y,z,Tz batch
#
import csv
import glob
import os
import shutil
import subprocess

from utils import csv_data


def create_scan_xyzTz_command(directory, row_data):
    """
    Constructs a well-formatted and robust command for scan_xyzTz.py.

    Args:
        directory: folder containing scan_xyzTz.py
        row_data
            mov_imgfile (str): Name of the reference image file.
            mov_seg_file (str): Name of the reference segmentation file.
            ref_imgfile (str): Name of the moving image file.
            ref_segfile (str): Name of the moving segmentation file.
            ref_scale_var (float): Scaling value limit of the reference image.
            xmi (int): Left bound pixel value of x search range.
            xma (int): Right bound pixel value of x search range.
            ymi (int): Top bound pixel value of y search range.
            yma (int): Bottom bound pixel value of y search range.
            zmi (int): Minimum z pxiel value of z search range.
            zma (int): Maximum z pxiel value of z search range.
            zstep (int): Step size for the z search.
            tz_center (float): Center value of the z angle search in degrees.
        Returns:
            str: The complete command string ready for execution.
    """

    command_args = [
        "python3",
        os.path.join(directory, "scan_xyzTz.py"),
        "--mov-imgfile",
        os.path.join(directory, row_data.mov_imgfile),
        "--mov-segfile",
        os.path.join(directory, row_data.mov_segfile),
        "--ref-imgfile",
        os.path.join(directory, row_data.ref_imgfile),
        "--ref-segfile",
        os.path.join(directory, row_data.ref_segfile),
        "--ref-scale-var",
        str(row_data.ref_scale_var),
        "--xmi",
        str(row_data.left_bound),
        "--xma",
        str(int(row_data.left_bound) + int(row_data.width)),
        "--ymi",
        str(row_data.top_bound),
        "--yma",
        str(int(row_data.top_bound) + int(row_data.height)),
        "--zmi",
        str(row_data.min_z),
        "--zma",
        str(row_data.max_z),
        "--zstep",
        str(row_data.z_step),
        "--angle",
        str(row_data.tz_center),
    ]

    return " ".join(command_args)


def main():
    """
    [Ubuntu]
    $ singularity exec --nv segreg_ubuntu20.sif python3.8 scan_xyzTz_batch.py 

    [CREATE]
    (base) k-number@erc-hpc-login2:~$ cat segreg.sh
    #!/bin/bash -l
    #SBATCH --output=/scratch_tmp/users/%u/%j.out
    #SBATCH --job-name=gpu
    #SBATCH --gres=gpu
    singularity exec --nv \
    segreg_ubuntu20.sif \
    python3.8 scan_xyzTz_batch.py
    (base) k-number@erc-hpc-login2:~$ sbatch -p gpu segreg.sh
    """

    directory = "./"
    paramfile = "create_figures.csv"

    with open(directory + paramfile, "r") as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            if row[0][0] != "#":
                results_dir = "results"

                row_data = csv_data.CSVData(*row)

                results_dir += (
                    f"_{row_data.mov_segfile[:-4]}_{row_data.ref_imgfile[:-4]}"
                )
                if not os.path.isdir(directory + results_dir):
                    os.mkdir(directory + results_dir)

                cmd = create_scan_xyzTz_command(directory, row_data)
                print(cmd)
                subprocess.run(cmd, shell=True)

                for tiffile in glob.glob("over*.tif"):
                    shutil.move(tiffile, os.path.join(directory, results_dir, tiffile))

                for logfile in glob.glob("align_[xT]*_[1-9]*.csv"):
                    shutil.move(logfile, os.path.join(directory + results_dir, logfile))


if __name__ == "__main__":
    main()
