import os
import shutil
from datetime import datetime
from pathlib import Path

from lfads_torch.run_model import run_model
from paths import runs_path

# ---------- OPTIONS -----------
PROJECT_STR = "lfads-torch-example"
DATASET_STR = "nlb_mc_rtt"
RUN_TAG = datetime.now().strftime("%y%m%d_%H%M%S") + "_exampleSingle"
# RUN_TAG = "231113_131336_exampleSingle"
RUN_DIR = Path(runs_path) / PROJECT_STR / DATASET_STR / RUN_TAG
OVERWRITE = False
# ------------------------------

# Overwrite the directory if necessary
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True,exist_ok=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Switch to the `RUN_DIR` and train the model
os.chdir(RUN_DIR)
run_model(
    # overrides={
    #     "datamodule": DATASET_STR,
    #     "model": DATASET_STR,
    # },
    config_path="../configs/single_few_shot_mc_rtt.yaml",
    # config_path="../configs/test.yaml",
    do_train=True,
    do_posterior_sample=False,
    do_fewshot_protocol=False,
    do_post_run_analysis=False,
)
