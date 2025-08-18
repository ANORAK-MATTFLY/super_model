import torch
from torch.utils import data
from ..models.interfaces.interface import IModel


class TrainingEngine:
    def __init__(self, model: IModel, dataloader: data.DataLoader, config: dict):
        self.model = model
        self.dataloader = dataloader
        self.config = config["training"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = self.config["device"]

    def train_one_epoch(self):
        self.model.train()
        self.model.to(self.device)
        total_loss = 0
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        print(f"Epoch Loss: {total_loss / len(self.dataloader):.4f}")
        return total_loss, total_loss

    # def run(self):
    #     print(f"--- Starting Training for model: {self.model.name} ---")
    #     for epoch in range(self.config["epochs"]):
    #         print(f"Epoch {epoch+1}/{self.config['epochs']}")
    #         self.train_one_epoch()
    #         # ... evaluation step would go here ...
    #     print("--- Training Finished ---")
    def run(self):
        # Start an MLflow run. MLflow automatically picks up the experiment name
        # from the tracking config or an environment variable.
        with mlflow.start_run() as mlf:
            print(f"--- Starting MLflow Run: {mlf.info.run_id} ---")
            print(f"--- Training model: {self.model.name} ---")

            # Log the entire Hydra config as a dictionary
            # This is CRITICAL for reproducibility
            flat_config = OmegaConf.to_container(
                self.config, resolve=True, throw_on_missing=True
            )
            mlflow.log_params(flat_config)

            self.model.to(self.device)
            for epoch in range(self.config["training"]["epochs"]):
                # ... (the training loop is the same) ...
                avg_loss, avg_accuracy = self.train_one_epoch()

                # Log metrics for each epoch
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                mlflow.log_metric("train_accuracy", avg_accuracy, step=epoch)

            # Log the trained model to the MLflow Model Registry
            mlflow.pytorch.log_model(
                pytorch_model=self.model,
                artifact_path="model",
                registered_model_name=self.model.name,  # Registers a new version under this name
            )
        print("--- Training Finished & Logged to MLflow ---")
