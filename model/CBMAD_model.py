import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Deepod.base_model import BaseDeepAD
from Deepod.utility import get_sub_seqs
from model.Mamba_block import Mamba_block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CBMAD(BaseDeepAD):

    def __init__(self, model_name="CBMAD", nb_feature=128, state_size=16,
                 num_layers=2, rms_norm=True, alpha=0.7,
                 seq_len=100, stride=1, epochs=100, batch_size=128, lr=1e-4,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):

        super(CBMAD, self).__init__(
            model_name=model_name, data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.seq_len = seq_len
        self.nb_feature = nb_feature
        self.state_size = state_size
        self.num_layers = num_layers
        self.rms_norm = rms_norm
        self.alpha = alpha

        self.net = CBMAD_model(nb_feature=nb_feature,
                               state_size=state_size,
                               num_layers=num_layers,
                               rms_norm=rms_norm,
                               ).to(self.device)

        return

    def fit(self, X, y=None):

        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)

        dataloader = DataLoader(seqs, batch_size=self.batch_size,
                                shuffle=False, drop_last=False)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.net.train()

        with tqdm(total=self.epochs, desc="Training Progress", unit="epoch") as pbar:
            for epoch in range(self.epochs):
                t1 = time.time()

                loss = self.training(dataloader, epoch, self.epochs)

                pbar.set_postfix({
                    "loss": f"{loss:.6f}",
                    "time": f"{time.time() - t1:.1f}s"
                })
                pbar.update(1)

                if self.verbose >= 1 and (epoch == 0 or (epoch + 1) % self.prt_steps == 0):
                    print(f'epoch{epoch + 1:3d}, '
                          f'training loss: {loss:.6f}, '
                          f'time: {time.time() - t1:.1f}s')

        self.decision_scores_ = self.decision_function(X)
        self.labels_ = self._process_decision_scores()  # in base model

        return

    def decision_function(self, X, return_rep=False):

        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
        dataloader = DataLoader(seqs, batch_size=self.batch_size,
                                shuffle=False, drop_last=False)

        anomaly_scores = self.inference(dataloader)

        print(f"anomaly_scores shape: {anomaly_scores.shape}")

        scores_final = np.mean(anomaly_scores, axis=(1))  # (n,)

        print(f"scores_final shape: {scores_final.shape}")

        scores_final_pad = np.hstack([0 * np.ones(X.shape[0] - scores_final.shape[0]), scores_final])

        print(f"scores_final_pad shape: {scores_final_pad.shape}")

        return scores_final_pad

    def training(self, dataloader, current_epoch, total_epoch):

        criterion = KL_Con_loss(epoch_start=1, epoch_end=total_epoch + 1, use_stop_grad=True).to(self.device)

        train_loss = []

        for ii, batch_x in enumerate(dataloader):

            self.optimizer.zero_grad()
            batch_x = batch_x.float().to(self.device)

            train_output, train_latent_s1_1, train_latent_s1_2 = self.net(batch_x)

            beta, recon_loss, kl_loss1 = criterion.forward(
                current_epoch,
                batch_x,
                train_output,
                train_latent_s1_1, train_latent_s1_2,
            )

            loss = (1 - beta) * recon_loss + beta * (kl_loss1)

            train_loss.append(loss.item())

            loss.backward()

            self.optimizer.step()

            if self.epoch_steps != -1 and ii > self.epoch_steps:
                break

        self.scheduler.step()
        return np.average(train_loss)

    def inference(self, dataloader):

        score = []

        loss_block = KL_Con_loss(epoch_start=1, epoch_end=2, use_stop_grad=True).to(self.device)

        with torch.no_grad():

            with tqdm(total=len(dataloader), desc="Inference Progress", unit="batch") as pbar:
                for batch_x in dataloader:  # test_set

                    batch_x = batch_x.float().to(self.device)

                    output_decoder, latent_s1_1, latent_s1_2 = self.net(batch_x)
                    anomaly_rec_score, anomaly_kl_score = loss_block.Calculate_anomaly_score(batch_x, output_decoder,
                                                                                             latent_s1_1, latent_s1_2, )


                    loss = (anomaly_rec_score ** (1-self.alpha)) * (anomaly_kl_score ** self.alpha)

                    loss = loss.detach().cpu().numpy()

                    score.append(loss)

                    pbar.update(1)

        anomaly_scores = np.concatenate(score, axis=0)

        print(f"Score shape: {anomaly_scores.shape}")

        return anomaly_scores

    def save_pt_model(self, path: str):

        if self.net is None:
            print("[Warning] No network to save.")
            return
        ckpt = {"model_state_dict": self.net.state_dict()}
        torch.save(ckpt, path)
        if self.verbose:
            print(f"[save_model] checkpoint saved at: {path}")

    def load_pt_model(self, path: str, map_location="cuda"):

        ckpt = torch.load(path, map_location=map_location)
        state_dict = ckpt["model_state_dict"]

        if self.net is None:
            self.net = CBMAD_model(nb_feature=self.nb_feature,
                                   state_size=self.state_size,
                                   num_layers=self.num_layers,
                                   rms_norm=self.rms_norm,
                                   ).to(self.device)

        self.net.load_state_dict(state_dict)

        if self.verbose:
            print(f"[load_pt_model] Model parameters loaded and state_dict restored from: {path}")

        return self

    def training_forward(self, batch_x, net, criterion):
        """define forward step in training"""
        return

    def inference_forward(self, batch_x, net, criterion):
        """define forward step in inference"""
        return

    def training_prepare(self, X, y):
        """define train_loader, net, and criterion"""
        return

    def inference_prepare(self, X):
        """define test_loader"""
        return


class MambaEncoder(nn.Module):
    def __init__(self, nb_feature, state_size, num_layers=2, rms_norm=True):
        super(MambaEncoder, self).__init__()

        self.nb_feature = nb_feature
        self.state_size = state_size
        self.num_layers = num_layers
        self.rms_norm = rms_norm

        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(in_channels=nb_feature, out_channels=nb_feature, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=nb_feature, out_channels=nb_feature, kernel_size=5, padding=2),
            nn.Conv1d(in_channels=nb_feature, out_channels=nb_feature, kernel_size=7, padding=3)
        ])

        # Stacked Mamba layers
        self.mamba_layers = nn.ModuleList([
            nn.Sequential(
                Mamba_block(
                    d_model=nb_feature,
                    d_state=state_size,
                    d_conv=4,
                    expand=2,
                    rms_norm=rms_norm
                ),
                nn.LayerNorm(nb_feature),
                nn.Linear(nb_feature, nb_feature * 2),
                nn.ReLU(),
                nn.Linear(nb_feature * 2, nb_feature)
            ) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(nb_feature)

    def forward(self, input_seq):

        cnn_out = input_seq
        for cnn_layer in self.cnn_layers:
            cnn_out = cnn_layer(cnn_out.transpose(1, 2)).transpose(1, 2)

        mamba_out = cnn_out

        for mamba_layer in self.mamba_layers:
            mamba_out = mamba_layer(mamba_out)

        output = self.layer_norm(mamba_out) + input_seq

        return output


class Bi_MambaEncoder(nn.Module):
    def __init__(self, nb_feature, state_size, num_layers=2, rms_norm=True):
        super(Bi_MambaEncoder, self).__init__()

        self.nb_feature = nb_feature
        self.state_size = state_size
        self.rms_norm = rms_norm
        self.num_layers = num_layers

        self.forward_mamba = MambaEncoder(
            nb_feature=nb_feature,
            state_size=state_size,
            num_layers=num_layers,
            rms_norm=rms_norm
        )

        self.backward_mamba = MambaEncoder(
            nb_feature=nb_feature,
            state_size=state_size,
            num_layers=num_layers,
            rms_norm=rms_norm
        )

    def forward(self, input_seq):
        reversed_seq = torch.flip(input_seq, dims=[1])

        forward = self.forward_mamba(input_seq)
        backward = self.backward_mamba(reversed_seq)
        backward = torch.flip(backward, dims=[1])

        return forward, backward


class MambaDecoder(nn.Module):
    def __init__(self, nb_feature, state_size, rms_norm=True):
        super(MambaDecoder, self).__init__()

        self.nb_feature = nb_feature
        self.state_size = state_size
        self.rms_norm = rms_norm

        self.mamba = Mamba_block(
            d_model=nb_feature,
            d_state=state_size,
            d_conv=4,
            expand=2,
            rms_norm=rms_norm
        )

        self.fc = nn.Sequential(
            nn.Linear(nb_feature, nb_feature * 2),
            nn.ReLU(),
            nn.Linear(nb_feature * 2, nb_feature)
        )

        combined_size = self.nb_feature * 2
        self.gate = GateMechanism(combined_size)

        self.d_mlp_1 = DualHiddenMLP(input_dim=self.nb_feature)

    def forward(self, forward_output, backward_output):
        combined_state = torch.cat((forward_output, backward_output), dim=2)

        gate_value_h = self.gate(combined_state)

        combined_state = gate_value_h * forward_output + (1 - gate_value_h) * backward_output

        output = self.mamba(combined_state)

        output = self.fc(output)

        latent_s1_1, latent_s1_2 = self.d_mlp_1(forward_output, backward_output)

        return output, latent_s1_1, latent_s1_2


class GateMechanism(nn.Module):
    def __init__(self, combined_size):
        super(GateMechanism, self).__init__()

        self.gate = nn.Sequential(
            nn.Linear(combined_size, combined_size // 2),
            nn.LeakyReLU(),
            nn.Linear(combined_size // 2, 1),  # you can either modify output nb_feature if you want feature-wise gating
            nn.Sigmoid(),
        )

    def forward(self, x):
        gate_values = self.gate(x)
        return gate_values


class Hidden_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64):
        super(Hidden_MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.activation2 = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)

        return x


class DualHiddenMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32):
        super(DualHiddenMLP, self).__init__()

        self.mlp1 = Hidden_MLP(input_dim, hidden_dim1, hidden_dim2)
        self.mlp2 = Hidden_MLP(input_dim, hidden_dim1, hidden_dim2)

    def forward(self, x1, x2):
        out1 = self.mlp1(x1)
        out2 = self.mlp2(x2)
        return out1, out2


import math

import torch
from torch import nn, optim
import torch.nn.functional as F


class KL_Con_loss(nn.Module):
    def __init__(self, epoch_start, epoch_end, use_epsilon=True, use_stop_grad=False):
        super(KL_Con_loss, self).__init__()
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end

        self.use_epsilon = use_epsilon
        self.use_stop_grad = use_stop_grad

        self.mse_criterion = nn.MSELoss(reduction='none')

    def sigmoid_weight(self, current_epoch):
        mid_point = (self.epoch_start + self.epoch_end) / 2
        steepness = 10 / (self.epoch_end - self.epoch_start)
        return 1 / (1 + math.exp(-steepness * (current_epoch - mid_point)))

    def Reconstruction_Loss(self, input_seq, output_seq):

        recon_loss = self.mse_criterion(input_seq, output_seq)
        return recon_loss.mean()

    def KL_Div(self, rpt1, rpt2):

        softmax_rpt1 = F.softmax(rpt1, dim=-1)
        softmax_rpt2 = F.softmax(rpt2, dim=-1)
        if self.use_epsilon:
            epsilon = 1e-8
            log_rpt1 = torch.log(softmax_rpt1 + epsilon)
            log_rpt2 = torch.log(softmax_rpt2 + epsilon)
        else:
            log_rpt1 = torch.log(softmax_rpt1)
            log_rpt2 = torch.log(softmax_rpt2)
        res = softmax_rpt1 * (log_rpt1 - log_rpt2)
        return torch.sum(res, dim=-1)

    def contrastive_loss(self, rpt1, rpt2):
        if self.use_stop_grad:

            rpt1_loss = self.KL_Div(rpt1, rpt2.detach()) + self.KL_Div(rpt2.detach(), rpt1)
            rpt2_loss = self.KL_Div(rpt2, rpt1.detach()) + self.KL_Div(rpt1.detach(), rpt2)

            return rpt1_loss + rpt2_loss

        else:

            return self.KL_Div(rpt1, rpt2) + self.KL_Div(rpt2, rpt1)

    def Calculate_anomaly_score(self, input_seq, output_seq, *rpts):

        recon_loss = self.mse_criterion(input_seq, output_seq)

        # if len(rpts) == 2: useless

        kl_loss1 = self.contrastive_loss(rpts[0], rpts[1])

        kl_loss = kl_loss1

        return torch.mean(recon_loss, dim=2), kl_loss


    def forward(self, current_epoch, input_seq, output_seq, *rpts):

        beta = self.sigmoid_weight(current_epoch=current_epoch)
        recon_loss = self.Reconstruction_Loss(input_seq, output_seq)
        beta = beta

        # if len(rpts) == 2: #useless

        kl_loss1 = self.contrastive_loss(rpts[0], rpts[1])

        return beta, recon_loss, kl_loss1.mean()



class CBMAD_model(nn.Module):

    def __init__(self, nb_feature, state_size, num_layers=2, official=True, rms_norm=True):
        super(CBMAD_model, self).__init__()

        self.official = official
        self.rms_norm = rms_norm
        self.state_size = state_size
        self.num_layers = num_layers

        self.encoder = Bi_MambaEncoder(nb_feature=nb_feature, state_size=state_size,
                                       num_layers=num_layers, rms_norm=rms_norm)
        self.decoder = MambaDecoder(nb_feature=nb_feature, state_size=state_size,
                                    rms_norm=rms_norm)

    def forward(self, input_seq):
        forward_output, backward_output = self.encoder(input_seq)

        output, latent_s1_1, latent_s1_2 = self.decoder(forward_output, backward_output)

        return output, latent_s1_1, latent_s1_2
