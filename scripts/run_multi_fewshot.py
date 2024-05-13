import os
import shutil
from datetime import datetime
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator
import json

from lfads_torch.run_model import run_model
from paths import runs_path

# ----------- Dataset-wise OPTIONS dict
options = {
    'mc_maze_5':{
        'DATASET_STR' : 'nlb_mc_maze',
        'config_path' : "../configs/multi_few_shot_original_heldout_mc_maze_5.yaml",
        'OLD_RUN_TAG' : '240318_144734_MultiFewshot',
        'experiment_json_path'  : 'experiment_state-2024-03-18_14-47-38.json',
        'model.dropout_rate': tune.uniform(0.0, 0.25), #narrowed down after first generation
    },
    'mc_maze_20':{
        'DATASET_STR' : 'nlb_mc_maze',
        'config_path' : "../configs/multi_few_shot_original_heldout_mc_maze.yaml",
        'OLD_RUN_TAG' : '240318_144734_MultiFewshot',
        'experiment_json_path'  : 'experiment_state-2024-03-18_14-47-38.json',
        'model.dropout_rate': tune.uniform(0.0, 0.25), #narrowed down after first generation
    },
    'mc_rtt_5':{
        'DATASET_STR' : 'nlb_mc_rtt',
        'config_path' : "../configs/multi_few_shot_original_heldout_mc_rtt.yaml",
        # 'OLD_RUN_TAG' : '240328_171607_MultiFewshot',
        # 'experiment_json_path' : 'experiment_state-2024-03-28_17-16-11.json',
        'OLD_RUN_TAG' : '240329_201611_MultiFewshot',
        'experiment_json_path' : 'experiment_state-2024-03-29_20-16-15.json',
        'model.dropout_rate': tune.uniform(0.0, 0.2),
    },
    'dmfc_rsg_5':{
        'DATASET_STR' : 'nlb_dmfc_rsg',
        'config_path' : "../configs/multi_few_shot_dmfc_rsg.yaml",
        # 'OLD_RUN_TAG' : '240401_164726_MultiFewshot',
        # 'experiment_json_path' : 'experiment_state-2024-04-01_16-47-30.json',
        'OLD_RUN_TAG' : '240402_111015_MultiFewshot',
        'experiment_json_path' : 'experiment_state-2024-04-02_11-10-19.json',
        'model.dropout_rate': tune.uniform(0.0, 0.3),
    }
}

# select_options = options['dmfc_rsg_5']
# select_options = options['mc_rtt_5']
select_options = options['mc_maze_20']

# ---------- OPTIONS -----------
PROJECT_STR = "lfads-torch-fewshot-benchmark"
DATASET_STR = select_options['DATASET_STR']
config_path = select_options['config_path']
num_samples = 50 #200  #160 
do_tune_run = True
RUN_TAG = datetime.now().strftime("%y%m%d_%H%M%S") + "_MultiFewshot"
# OLD_RUN_TAG = '231110_002643_MultiFewshot'
# experiment_json_path = 'experiment_state-2023-11-10_00-26-47.json'

# load small mc_maze test checkpoints
# num_samples = 1

#### first set of 200 models #####
# OLD_RUN_TAG = '240314_172554_MultiFewshot' 
# experiment_json_path = 'experiment_state-2024-03-14_17-25-57.json'  #'experiment_state-2024-03-14_14-14-57.json'

#### second set of 200 models #####
OLD_RUN_TAG = select_options['OLD_RUN_TAG']
experiment_json_path = select_options['experiment_json_path']
load_old_checkpoints = True

RUN_DIR     = Path(runs_path) / PROJECT_STR / DATASET_STR / RUN_TAG
OLD_RUN_DIR = None

# ------------------------------

# Set the mandatory config overrides to select datamodule and model
mandatory_overrides = {
    # "datamodule": DATASET_STR,
    # "model": DATASET_STR,
    "logger.wandb_logger.project": PROJECT_STR,
    "logger.wandb_logger.tags.1": DATASET_STR,
    "logger.wandb_logger.tags.2": RUN_TAG,
}


trial_ids = None

if load_old_checkpoints:
    OLD_RUN_DIR = Path(runs_path) / PROJECT_STR / DATASET_STR / OLD_RUN_TAG
    experiment_json_path_full = OLD_RUN_DIR / experiment_json_path
    print(experiment_json_path_full)
    if os.path.exists(experiment_json_path_full):
        with open(experiment_json_path_full) as f:
            experiment_data = json.load(f)
        trial_ids = [json.loads(experiment)['trial_id'] for experiment in experiment_data['checkpoints']]
    else:
        raise FileNotFoundError()


RUN_DIR.mkdir(parents=True,exist_ok=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)

if do_tune_run:
    # Run the hyperparameter search
    tune.run(
        tune.with_parameters(
            run_model,
            config_path=config_path,
            do_train=False,
            do_posterior_sample=False,
            do_fewshot_protocol=False,
            do_post_run_analysis=False,
            do_nlb_fewshot = False,
            do_nlb_fewshot2 = True,
            variant = 'mc_maze_20',
            run_dir = OLD_RUN_DIR,
            trial_ids = trial_ids,
            load_best = True
        ),
        # metric="valid/recon_smth",  removed for loading checkpoints for analysis
        # mode="min",
        name=RUN_DIR.name,
        config={
            **mandatory_overrides,
            # "dropout_target" : tune.choice([
            #     'lfads_torch.modules.augmentations.CoordinatedDropout',
            #     'lfads_torch.modules.augmentations.CoordinatedDropoutChannelWise',
            # ]),
            "cd_rate" : tune.uniform(0.05, 0.4),
            # "model.dropout_rate" : tune.uniform(0.0, 0.6),
            "model.dropout_rate" : select_options['model.dropout_rate'],
            "model.kl_co_scale"  : tune.loguniform(1e-6, 1e-4),
            "model.kl_ic_scale"  : tune.loguniform(1e-6, 1e-3),
            "model.l2_gen_scale" : tune.loguniform(1e-4, 1e0),
            "model.l2_con_scale" : tune.loguniform(1e-4, 1e0),
        },
        resources_per_trial=dict(cpu=3, gpu=0.5),
        num_samples=num_samples,
        local_dir=RUN_DIR.parent,
        search_alg=BasicVariantGenerator(random_state=0),
        scheduler=FIFOScheduler(),
        verbose=1,
        progress_reporter=CLIReporter(
            metric_columns=["valid/recon_smth", "cur_epoch"],
            sort_by_metric=True,
        ),
        trial_dirname_creator=lambda trial: str(trial),
    )