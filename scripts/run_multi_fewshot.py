import shutil
from datetime import datetime
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from lfads_torch.run_model import run_model
from paths import runs_path

# ---------- OPTIONS -----------
PROJECT_STR = "lfads-torch-fewshot-benchmark"
DATASET_STR = "nlb_mc_maze"
# RUN_TAG = datetime.now().strftime("%y%m%d_%H%M%S") + "_MultiFewshot"
RUN_TAG = '231109_164714_MultiFewshot'
RUN_DIR = Path(runs_path) / PROJECT_STR / DATASET_STR / RUN_TAG
# ------------------------------

# Set the mandatory config overrides to select datamodule and model
mandatory_overrides = {
    "datamodule": DATASET_STR,
    "model": DATASET_STR,
    "logger.wandb_logger.project": PROJECT_STR,
    "logger.wandb_logger.tags.1": DATASET_STR,
    "logger.wandb_logger.tags.2": RUN_TAG,
}
RUN_DIR.mkdir(parents=True,exist_ok=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Run the hyperparameter search
tune.run(
    tune.with_parameters(
        run_model,
        config_path="../configs/multi_few_shot.yaml",
        do_train=False,
        do_posterior_sample=False,
        do_fewshot_protocol=True,
    ),
    metric="valid/recon_smth",
    mode="min",
    name=RUN_DIR.name,
    config={
        **mandatory_overrides,
        "dropout_target" : tune.choice([
            'lfads_torch.modules.augmentations.CoordinatedDropout',
            'lfads_torch.modules.augmentations.CoordinatedDropoutChannelWise',
        ]),
        "cd_rate" : tune.uniform(0.05, 0.4),
        "model.dropout_rate": tune.uniform(0.0, 0.6),
        "model.kl_co_scale": tune.loguniform(1e-6, 1e-4),
        "model.kl_ic_scale": tune.loguniform(1e-6, 1e-3),
        "model.l2_gen_scale": tune.loguniform(1e-4, 1e0),
        "model.l2_con_scale": tune.loguniform(1e-4, 1e0),
    },
    resources_per_trial=dict(cpu=3, gpu=0.5),
    num_samples=60,
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
