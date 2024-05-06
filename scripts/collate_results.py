import os
import pandas as pd
from pathlib import Path

def read_concat_results(root_dir):
    # Initialize an empty DataFrame to store concatenated results
    concatenated_results = pd.DataFrame()

    # Iterate through the root directory and its subdirectories
    for subdir, _, files in os.walk(root_dir):
        # Check if any file named 'results.csv' exists in the current directory
        if 'results.csv' in files:
            # Form the full path of the 'results.csv' file
            results_file_path = os.path.join(subdir, 'results.csv')
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(results_file_path,index_col=0)
            
            # Concatenate the current DataFrame with the overall concatenated results
            concatenated_results = pd.concat([concatenated_results, df], ignore_index=True)

    return concatenated_results

def main():
    # Root directory containing 'results.csv' files and its subdirectories
    root_directory = Path('/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240501_030533_MultiFewshot')

    # Read and concatenate 'results.csv' files
    concatenated_results = read_concat_results(root_directory)

    # Save the concatenated results to a new CSV file
    concatenated_results.to_csv(root_directory / 'concatenated_results.csv')

    print("Concatenated results saved successfully!")

if __name__ == "__main__":
    main()
