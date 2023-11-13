import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl


class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model architecture here
        self.fc = nn.Linear(784, 10)  # Example linear layer for illustration

    def forward(self, x):
        # Define the forward pass
        return self.fc(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        # Skip updating model parameters for this example
        # This essentially skips the backward pass and optimizer step
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)

        # Log the loss for information
        self.log('train_loss', loss)

        # Return the loss to indicate that the training step is complete
        return loss

    def configure_optimizers(self):
        # Define your optimizer, but it won't be used since we're skipping the optimizer step
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Dummy DataLoader for illustration purposes
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Instantiate the Lightning Model
model = MyModel()

# Instantiate the Lightning Trainer
trainer = pl.Trainer(max_epochs=10, gpus=1)

# Train the model without updating parameters
trainer.fit(model, train_loader)
