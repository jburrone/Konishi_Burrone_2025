import argparse
import csv
import glob
import os

from utils import csv_data


def filter_data(data, z):
    """Filters data based on the 'z_segreg' value.

    Args:
        data (list): A list of dictionaries, each representing a data point.
        z (int): The target 'z_segreg' value.

    Returns:
        A filtered list of dictionaries.
    """

    return [
        {
            "z_segreg": int(item["z_segreg"]),
            "y_offmov": int(item["y_offmov"]),
            "y_segreg": int(item["y_segreg"]),
            "x_offmov": int(item["x_offmov"]),
            "x_segreg": int(item["x_segreg"]),
            "tz_segreg": float(item["tz_segreg"]),
            "sx_segreg": float(item["sx_segreg"]),
            "sy_segreg": float(item["sy_segreg"]),
            "mi": float(item["mi"]),
            "opixmatch": float(item["opixmatch"]),
            "jaccard": float(item["jaccard"]),
            "omatch": float(item["omatch"]),
            "ce": float(item["ce"]),
            "all": float(item["all"]),
        }
        for item in data
        if item["z_segreg"] == z
    ]


def generate_viewer_script(top_results, os_name, bashfile, search):
    """Generates a Bash script to open the top results.

    Args:
        top_results (list): A list of dictionaries containing result information.
        os_name (str): The name of the operating system.
        bashfile (str): The path to the output Bash script file.
        search (str): search axis. should be 'xyzTz', 'Ty', 'Tx'.
    """

    try:
        with open(bashfile, "w") as f:
            if os_name == "nt":
                f.write(
                    r"%SystemRoot%\System32\rundll32.exe \"%ProgramFiles%\Windows Photo Viewer\PhotoViewer.dll\", ImageView_Fullscreen\n"
                )

            for result in top_results:
                if search == "xyzTz":
                    filename = f"overlap_reg_z{result['z_segreg']:02d}_y{result['y_offmov']:03d}_x{result['x_offmov']:03d}_tz{result['tz_segreg']:04.1f}_mi{result['mi']:0.3f}_opixmatch{result['opixmatch']:.2f}_jaccard{result['jaccard']:.2f}_omatch{result['omatch']:.2f}_ce{result['ce']:.2f}.tif"
                elif search == "Ty":
                    filename = f"overlap_reg_ty{result['ty_segreg']:04.1f}_mi{result['mi']:0.3f}_opixmatch{result['opixmatch']:.2f}_jaccard{result['jaccard']:.2f}_omatch{result['omatch']:.2f}_ce{result['ce']:.2f}.tif"
                elif search == "Tx":
                    filename = f"overlap_reg_tx{result['tx_segreg']:04.1f}_mi{result['mi']:0.3f}_opixmatch{result['opixmatch']:.2f}_jaccard{result['jaccard']:.2f}_omatch{result['omatch']:.2f}_ce{result['ce']:.2f}.tif"

                f.write(f"# {filename}\n")
                if os_name == "nt":
                    write_image_opening_command(f, filename, os_name)
                elif os_name == "posix":
                    write_image_opening_command(f, os.path.join(results_dir, filename), os_name)
    except IOError as e:
        print(f"Error writing to file: {e}")


def generate_txtfile(data, metric, output_file, search):
    """Writes selected data from a list of dictionaries to a text file.

    Args:
        data (list): A list of dictionaries containing the data.
        metric (str): The key of the metric to be written to the file.
        output_file (str): The path to the output file.
        search (str): search axis. should be 'xyzTz', 'Ty', 'Tx'.
    """

    try:
        with open(output_file, "w", newline="") as file:
            writer = csv.writer(file)
            for item in data:
                if search == "xyzTz":
                    writer.writerow([item["z_segreg"], item[metric]])
                elif search == "Ty":
                    writer.writerow([item["ty_segreg"], item[metric]])
                elif search == "Tx":
                    writer.writerow([item["tx_segreg"], item[metric]])
    except IOError as e:
        print(f"Error writing to file: {e}")


def read_csv_data(csv_file, search):
    """Reads data from a CSV file and filters rows based on a condition.

    Args:
        csv_file (str): The path to the CSV file.
        search (str): search axis. should be 'xyzTz', 'Ty', 'Tx'.

    Returns:
        A list of dictionaries, where each dictionary represents a row of data.
    """

    data = []
    try:
        with open(csv_file, "r") as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                if search == "xyzTz":
                    row_data = csv_data.CSVData_xyzTz(*row)
                    if float(row_data.mi) > 0:
                        data.append(
                            {
                                "z_segreg": int(row_data.z_segreg),
                                "y_offmov": int(row_data.y_offmov),
                                "y_segreg": int(row_data.y_segreg),
                                "x_offmov": int(row_data.x_offmov),
                                "x_segreg": int(row_data.x_segreg),
                                "tz_segreg": float(row_data.tz_segreg),
                                "sx_segreg": float(row_data.sx_segreg),
                                "sy_segreg": float(row_data.sy_segreg),
                                "mi": float(row_data.mi),
                                "opixmatch": float(row_data.opixmatch),
                                "jaccard": float(row_data.jaccard),
                                "omatch": float(row_data.omatch),
                                "ce": float(row_data.ce),
                                "all": float(row_data.all),
                            }
                        )
                elif search == "Ty":
                    row_data = csv_data.CSVData_Ty(*row)
                    data.append(
                        {
                            "ty_segreg": float(row_data.ty_segreg),
                            "y_segreg": int(row_data.y_segreg),
                            "x_segreg": int(row_data.x_segreg),
                            "sx_segreg": float(row_data.sx_segreg),
                            "sy_segreg": float(row_data.sy_segreg),
                            "mi": float(row_data.mi),
                            "opixmatch": float(row_data.opixmatch),
                            "jaccard": float(row_data.jaccard),
                            "omatch": float(row_data.omatch),
                            "ce": float(row_data.ce),
                            "all": float(row_data.all),
                        }
                    )
                elif search == "Tx":
                    row_data = csv_data.CSVData_Tx(*row)
                    data.append(
                        {
                            "tx_segreg": float(row_data.tx_segreg),
                            "y_segreg": int(row_data.y_segreg),
                            "x_segreg": int(row_data.x_segreg),
                            "sx_segreg": float(row_data.sx_segreg),
                            "sy_segreg": float(row_data.sy_segreg),
                            "mi": float(row_data.mi),
                            "opixmatch": float(row_data.opixmatch),
                            "jaccard": float(row_data.jaccard),
                            "omatch": float(row_data.omatch),
                            "ce": float(row_data.ce),
                            "all": float(row_data.all),
                        }
                    )
                # print(row_data)

    except FileNotFoundError:
        print(f"CSV file not found: {csv_file}")
    except csv.Error as e:
        print(f"Error reading CSV file: {e}")

    return data


def print_top_results(data, search):
    """
    Prints top results.

    Args:
        data (list): A list of dictionaries, each representing a data point.
        search (str): search axis. should be 'xyzTz', 'Ty', 'Tx'.
    """
    print(f"# MI: {data['mi']:0.3f}")

    if search == "xyzTz":
        print(f"x_offmov: {data['x_offmov']}")
        print(f"y_offmov: {data['y_offmov']}")
        print(f"tz_segreg: {data['tz_segreg']:0.2f}")
    elif search == "Ty":
        print(f"ty_segreg: {data['ty_segreg']:0.2f}")
    elif search == "Tx":
        print(f"tx_segreg: {data['tx_segreg']:0.2f}")

    print(f"x_segreg: {data['x_segreg']}")
    print(f"y_segreg: {data['y_segreg']}")
    if search == "xyzTz":
        print(f"z_segreg: {data['z_segreg']}")
    print(f"sx_segreg: {data['sx_segreg']:0.2f}")
    print(f"sy_segreg: {data['sy_segreg']:0.2f}")
    return


def write_image_opening_command(f, filename, os_name):
    """
    Writes the appropriate command to open the image based on the OS.

    Args:
        f: The file object to write to.
        filename: The filename of the image.
        os_name: The name of the operating system.
    """

    if os_name == "posix":
        f.write(f"eog {filename}\n")
    elif os_name == "nt":
        f.write(f"{filename}\n")
        f.write("PAUSE\n")


def parse_args():

    parser = argparse.ArgumentParser(
        prog="sort_segreg.py",
        description="Sort registration list based on metric value.",
        add_help=True,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory of image registration result",
    )
    parser.add_argument("-z", default=-1, type=int, help="Search z number. -1 for all z plane")
    parser.add_argument(
        "--axis", default="xyzTz", type=str, help="Search parameters: xyzTz, Ty, Tx"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    """
    Local PC:
    singularity exec --nv segreg_ubuntu20.sif python3.8 sort_registration.py \
        --results-dir results_373_231009_SoRa_Blk_20x1x_seg_373_231115_AX_Sect1_25xOil_img_roi1 \
        --axis Tx

    or
    HPC:
    srun -p gpu --gres gpu --constraint a100_40g --pty /bin/bash
    singularity exec --nv --bind folder_name:/container-dir \
        segreg_ubuntu20.sif \
        python3.8 /container-dir/sort_registration.py \
        --results-dir results_373_231009_SoRa_Blk_20x1x_seg_373_231115_AX_Sect1_25xOil_img_roi1 \
        --axis Tx

    or
    sbatch -p gpu align_all.sh
    """
    # TODO: run this script on HPC
    args = parse_args()

    results_dir = args.results_dir
    os_name = os.name

    csvfilepaths = glob.glob(os.path.join(results_dir + f"/*{args.axis}*.csv"))

    data = []
    for csvfilepath in csvfilepaths:
        data.extend(read_csv_data(csvfilepath, args.axis))

    if args.z == -1:
        filtered_data = data
    else:
        filtered_data = filter_data(data, args.z)

    # metric_candidates = ['mi', 'opixmatch', 'omatch', 'jaccard', 'ce']
    metric_candidates = ["mi"]

    for metric in metric_candidates:
        txtfile = f"{results_dir}_{metric}_{args.axis}.txt"
        generate_txtfile(filtered_data, metric, txtfile, args.axis)

        bashfile = f"display_matches_{os.path.split(results_dir[:-1])[-1]}_{metric}_{args.axis}.sh"
        if os_name == "nt":
            bashfile = bashfile.replace(".sh", ".bat")

        top3_results = sorted(filtered_data, key=lambda x: x[metric], reverse=True)[:3]
        generate_viewer_script(top3_results, os_name, bashfile, args.axis)
        print_top_results(top3_results[0], args.axis)

        print(f"Output {txtfile}")
        print(f"Output {bashfile}")
