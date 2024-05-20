# import os
# import pandas as pd
from pathlib import Path
from nlb_tools.collate_results import read_concat_results

# def read_concat_results(root_dir,endswith='results.csv'):
#     # Initialize an empty DataFrame to store concatenated results
#     concatenated_results = pd.DataFrame()

#     # Iterate through the root directory and its subdirectories
#     for subdir, _, files in os.walk(root_dir):
#         # Check if any file named 'results.csv' exists in the current directory
#         for f in files:
#             if f.endswith(endswith):                    
#                 # Form the full path of the 'results.csv' file
#                 results_file_path = os.path.join(subdir, f)
                
#                 # Read the CSV file into a DataFrame
#                 df = pd.read_csv(results_file_path,index_col=0)
                
#                 # Concatenate the current DataFrame with the overall concatenated results
#                 concatenated_results = pd.concat([concatenated_results, df], ignore_index=True)

#     return concatenated_results

def main():
    # Root directory containing 'results.csv' files and its subdirectories
    # root_directory = Path('/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_dmfc_rsg/240501_030533_MultiFewshot')
    # root_directory = Path('/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240511_141307_MultiFewshot')
    # root_directory = Path('/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240513_193832_MultiFewshot')
    # root_directory = Path('/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240514_000059_MultiFewshot')
    # root_directory = Path('/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240514_220948_MultiFewshot')
    # root_directory = Path('/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240515_075022_MultiFewshot')
    # root_directory = Path('/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240517_121846_MultiFewshot')
    # root_directory = Path('/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240518_134433_MultiFewshot')
    # root_directory = Path('/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240519_184438_MultiFewshot')
    root_directory = Path('/home/kabird/lfads-torch-runs/lfads-torch-fewshot-benchmark/nlb_mc_maze/240519_194718_MultiFewshot') # 128 and 256 all models
    
    

    # Read and concatenate 'results.csv' files
    # concatenated_results = read_concat_results(root_directory,endswith='results_logrates-as-latents.csv')
    concatenated_results = read_concat_results(root_directory,endswith='results.csv')

    # Save the concatenated results to a new CSV file
    concatenated_results.to_csv(root_directory / 'concatenated_results.csv')

    print("Concatenated results saved successfully!")

if __name__ == "__main__":
    main()
