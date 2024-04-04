import hydra
from hydra.utils import call, instantiate


@hydra.main(version_base='1.3', config_path='configs', config_name='mc_maze_05.yaml')
def main(cfg):
    dataset = instantiate(cfg.load_dataset.dataset)

if __name__=='__main__':
    main()