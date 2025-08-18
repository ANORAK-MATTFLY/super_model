import hydra
from omegaconf import DictConfig, OmegaConf
from ..system.factory import get_model
from ..system.engine import TrainingEngine

# Assume a get_dataloader function exists
from ..data.dataloaders import get_dataloader


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))

    # 1. Instantiate model using the factory and config
    # cfg.model is the sub-config from either resnet.yaml or vit.yaml
    model = get_model(cfg.model)

    # 2. Instantiate dataloader
    # This would likely use another factory pattern for datasets
    train_loader = get_dataloader(cfg.dataset, cfg.training)

    # 3. Instantiate and run the engine
    engine = TrainingEngine(model, train_loader, cfg)
    engine.run()


if __name__ == "__main__":
    train()
