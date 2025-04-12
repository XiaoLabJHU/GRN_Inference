"""
This script is intended to generate X replicates (default = 10) for each of the DREAM (3/4)
networks to check inference algorithm performance.
The pipeline works as follows:
1) For each network kinetic model file (.xml)
Repeat X times:
2) GNW: generate expression data (it generates it in the app folder due to a bug)
3) Make a new subdirectory for replicate_X (including all parent directories)
4) Move expression data files to the network subdirectory
5) Combine Steady-State expression data files to one file

Written by: Lior Shachaf
2020-11-09

Example usage:
python3 Generate_replicates_for_network_inference.py -r 10 -o ~/Data/ --path_to_gnw ~/Code/genenetweaver/ dream3
"""

import argparse
import os
import shutil  # Import shutil for moving files across devices
import subprocess


def generate_replicates(dream_version, replicates, output_path, path_to_gnw):
    """
    Generate replicates for each network XML file in the specified directory.

    Parameters:
    dream_version (str): The DREAM version (dream3 or dream4).
    replicates (int): Number of replicates to generate.
    output_path (str): Path to store the generated data.
    path_to_gnw (str): Path to the GeneNetWeaver (GNW) installation.

    Returns:
    None
    """
    # Define paths based on user input
    path_to_gnw = os.path.abspath(os.path.expanduser(path_to_gnw))
    path_to_dreamX_networks = os.path.join(path_to_gnw, f'./src/ch/epfl/lis/networks/{dream_version}')
    output_path = os.path.abspath(os.path.join(os.path.expanduser(output_path), dream_version))

    # Iterate over all network XML files
    for file in os.listdir(path_to_dreamX_networks):
        if file.endswith('.xml'):
            print(file)
            filename = os.path.splitext(file)[0]

            # Generate replicates
            for replicate in range(1, replicates + 1):
                # Create target directory for the replicate
                replicate_dir = os.path.join(output_path, filename, f'rep_{replicate}')
                os.makedirs(replicate_dir, exist_ok=True)

                # Run the GNW simulation
                subprocess.run(['java', '-jar', os.path.join(path_to_gnw, 'gnw-3.1.2b.jar'),
                                '--simulate', '-c', 'settings.txt', '--input-net', os.path.join(path_to_dreamX_networks, file)])

                # Move generated files to the target directory
                for generated_file in os.listdir(os.getcwd()):
                    if generated_file.startswith(filename):
                        shutil.move(os.path.join(os.getcwd(), generated_file), os.path.join(replicate_dir, generated_file))

    print("\nReplicates generation done\n")


def combine_steady_state_data(dream_version, output_path):
    """
    Combine Steady-State expression data (wildtype, multifactorial, knockdowns, knockouts, dualknockouts) into one file.

    Parameters:
    dream_version (str): The DREAM version (dream3 or dream4).
    output_path (str): Path to store the combined data.

    Returns:
    None
    """
    # Change to specific DREAM data folder containing the different network folders
    path_to_data = os.path.abspath(os.path.join(os.path.expanduser(output_path), dream_version))
    os.chdir(path_to_data)

    data_type_list = ["wildtype", "multifactorial", "knockdowns", "knockouts", "dualknockouts"]

    for network_name in os.listdir():
        if os.path.isdir(network_name):
            os.chdir(network_name)

            for replicate in os.listdir():
                if os.path.isdir(replicate):
                    os.chdir(replicate)

                    output_file_name = f"{network_name}_SS_all.tsv"
                    with open(output_file_name, "w") as output_file:
                        for data_type in data_type_list:
                            input_file = f"{network_name}_{data_type}.tsv"
                            with open(input_file, "r") as in1:
                                data1 = in1.readlines()

                            for line in data1:
                                if "G1" not in line:
                                    output_file.write(line)

                    os.chdir('..')
            os.chdir('..')

    print("Combine steady-states done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate replicates and combine steady-state data for DREAM networks.")
    parser.add_argument("dream_version", choices=["dream3", "dream4"], help="Specify the DREAM version (dream3 or dream4).")
    parser.add_argument("-r", "--replicates", type=int, default=10, help="Number of replicates to generate. Default=10")
    parser.add_argument("-o", "--output_path", type=str, default=".", help="Output path to store the data. Default=.")
    parser.add_argument("--path_to_gnw", type=str, default="~/genenetweaver/", help="Path to GeneNetWeaver (GNW) installation.")

    args = parser.parse_args()

    generate_replicates(args.dream_version, args.replicates, args.output_path, args.path_to_gnw)
    combine_steady_state_data(args.dream_version, args.output_path)
