import warnings
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from mineclip import MineCLIP


class MineCLIPWrapper(L.LightningModule):
    def __init__(self, ckpt_path):
        super().__init__()
        self.mineclip = MineCLIP(
            arch="vit_base_p16_fz.v2.t2",
            hidden_dim=512,
            image_feature_dim=512,
            mlp_adapter_spec="v0-2.t0",
            pool_type="attn.d2.nh8.glusw",
            resolution=(160, 256),
        )
        self.mineclip.load_ckpt(ckpt_path, strict=True)

    def forward(self, x):
        return self.mineclip.encode_video(x)


class ShortTermGoalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc_long = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, l):
        l = F.relu(self.fc_long(l))
        x = F.relu(self.fc1(x))
        x = x + l
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_var = self.fc_var(x)
        return mean, log_var


class ShortTermGoalDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc_long = nn.Linear(output_dim, hidden_dim)
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, l):
        l = F.relu(self.fc_long(l))
        x = F.relu(self.fc1(x))
        x = x + l
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ShortTermGoalCVAE(L.LightningModule):
    def __init__(
        self,
        mineclip_path,
        clip_dim,
        hidden_dim,
        latent_dim,
        kl_weight=1e-2,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mineclip = MineCLIPWrapper(mineclip_path)
        self.mineclip.freeze()
        self.encoder = ShortTermGoalEncoder(clip_dim, hidden_dim, latent_dim)
        self.decoder = ShortTermGoalDecoder(latent_dim, hidden_dim, clip_dim)
        self.strict_loading = False

    def forward(self, x, l):
        x = self.mineclip(x)
        l = self.mineclip(l)
        mean, log_var = self.encoder(x, l)
        z = self.reparameterize(mean, log_var)
        y = self.decoder(z, l)
        return x, y, mean, log_var

    def training_step(self, batch, batch_idx):
        x, l = batch
        x, y, mean, log_var = self(x, l)
        loss = self.calc_loss(x, y, mean, log_var)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, l = batch
        x, y, mean, log_var = self(x, l)
        loss = self.calc_loss(x, y, mean, log_var)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def state_dict(self, **kwargs):
        return {
            k: v for k, v in super().state_dict(**kwargs).items() if "mineclip" not in k
        }

    def calc_loss(self, x, y, mean, log_var):
        recon_loss = F.mse_loss(y, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = recon_loss + self.hparams.kl_weight * kl_loss
        return loss

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, x, l):
        x = self.mineclip(x)
        l = self.mineclip(l)
        mean, log_var = self.encoder(x, l)
        z = self.reparameterize(mean, log_var)
        return z


class ShortTermGoalModel(L.LightningModule):
    def __init__(self, encoder_path, latent_dim, hidden_dim, n_layers, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.encoder = ShortTermGoalCVAE.load_from_checkpoint(encoder_path)
        self.encoder.freeze()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.strict_loading = False

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        for layer in self.layers:
            x = x + layer(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, l, y = batch
        x = self.encoder.encode(x, l)
        y = self.encoder.encode(y, l)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, l, y = batch
        x = self.encoder.encode(x, l)
        y = self.encoder.encode(y, l)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def state_dict(self, **kwargs):
        return {
            k: v for k, v in super().state_dict(**kwargs).items() if "encoder" not in k
        }
