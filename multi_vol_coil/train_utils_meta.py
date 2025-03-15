import os
from pathlib import Path
from typing import Optional
from itertools import chain

import fastmri
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils import *
import copy
from torch.optim import SGD, Adam, AdamW
from fastmri.data.transforms import tensor_to_complex_np, to_tensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

OPTIMIZER_CLASSES = {
    "Adam": Adam,
    "AdamW": AdamW,
    "SGD": SGD,
}

class Trainer:
    def __init__(
        self, dataloader, embeddings_vol, embeddings_coil, embeddings_start_idx, model, loss_fn, optimizer, scheduler, config
        ) -> None:
        self.device = torch.device(config["device"])
        self.n_epochs = config["n_epochs"]

        self.dataloader = dataloader
        self.embeddings_vol, self.embeddings_coil = embeddings_vol.to(self.device), embeddings_coil.to(self.device)
        
        self.start_idx = embeddings_start_idx.to(self.device)
        self.model = model.to(self.device)
        

        # If stateful loss function, move its "parameters" to `device`.
        if hasattr(loss_fn, "to"):
            self.loss_fn = loss_fn.to(self.device)
        else:
            self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.log_interval = config["log_interval"]
        self.inference_mode = config["dataset"]["with_mask"]
        self.checkpoint_interval = config["checkpoint_interval"]
        self.path_to_out = Path(config["path_to_outputs"])
        self.timestamp = config["timestamp"]
        self.config = config
        self.writer = SummaryWriter(self.path_to_out / self.timestamp)

        # Ground truth (used to compute the evaluation metrics).
        self.ground_truth = []
        self.kspace_gt = []
        for vol_id in self.dataloader.dataset.metadata.keys():
            file = self.dataloader.dataset.metadata[vol_id]["file"]
            with h5py.File(file, "r") as hf:
                self.ground_truth.append(
                    hf["reconstruction_rss"][()][: config["dataset"]["n_slices"]]
                )
                self.kspace_gt.append(
                    to_tensor(preprocess_kspace(hf["kspace"][()][: config["dataset"]["n_slices"]]
                )
                    )
                    
                )
        # Scientific and nuissance hyperparameters.
        self.hparam_info = config["hparam_info"]
        self.hparam_info["embedding_vol_dim"] = self.embeddings_vol.weight.shape[1]
        self.hparam_info["embedding_coil_dim"] = self.embeddings_coil.weight.shape[1]

        # Evaluation metrics for the last log.
        self.last_nmse = [0] * len(self.dataloader.dataset.metadata)
        self.last_psnr = [0] * len(self.dataloader.dataset.metadata)
        self.last_ssim = [0] * len(self.dataloader.dataset.metadata)

    ###########################################################################
    ###########################################################################
    ###########################################################################

    def train(self):
        """Train the model across multiple epochs and log the performance."""
        empirical_risk = 0
        for epoch_idx in range(self.n_epochs):
            empirical_risk = self._train_one_epoch(epoch_idx)

            print(f"EPOCH {epoch_idx}    avg loss: {empirical_risk}\n")
            self.writer.add_scalar("Loss/train", empirical_risk, epoch_idx)

            if (epoch_idx + 1) % self.log_interval == 0:
                self._log_performance(epoch_idx)
                self._log_weight_info(epoch_idx)

            if (epoch_idx + 1) % self.checkpoint_interval == 0:
                # Takes ~3 seconds.
                self._save_checkpoint(epoch_idx)

        self._log_information(empirical_risk)
        self.writer.close()
        
    
    def add_noise_model(self, model_temp):
        for _, tensor in model_temp.named_parameters():
            tshape = torch.tensor(tensor.shape, dtype=float)
            scale = torch.norm(tshape)

            noise = torch.normal(0, torch.std(tensor).item(), tensor.size(), device=self.device)
            tensor.data.add_(noise)
            
        return model_temp
    
    def add_noise_embedding(self, embedding):
        for vect in embedding.parameters():
            tshape = torch.tensor(vect.size(), dtype = float)
            scale = torch.norm(tshape)
            
            noise = torch.normal(0, torch.std(vect).item(), size=vect.size(), device=self.device)
            vect.data.add_(noise)
            
        return embedding
    
    def check_amount_noise(self, model_noise):
        noise_total = []
        for (_, tensor1), (_, tensor2) in zip(self.model.named_parameters(), model_noise.named_parameters()):
            noise_total.append(torch.norm(tensor1.data-tensor2.data).detach().cpu())
        return np.mean(noise_total)
        
    def _train_one_epoch(self, epoch_idx):
        # Also known as "empirical risk".
        avg_loss = 0.0
        n_obs = 0
        self.model.train()
        
        for batch_idx, (inputs, _, targets) in enumerate(self.dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Get the index for the coil latent embedding
            coords = inputs[:, 1:-1]
            vol_ids = inputs[:,0].long()
            coil_ids = inputs[:,-1].long() 
            
            # ## ADD NOISE TO EMBEDDINGS
            # embeddings_vol_temp = copy.deepcopy(self.embeddings_vol)
            # embeddings_coil_temp = copy.deepcopy(self.embeddings_coil)
            
            # embeddings_vol_noise = self.add_noise_embedding(embeddings_vol_temp)
            # embeddings_coil_noise = self.add_noise_embedding(embeddings_coil_temp)
            
            # noise_coil = torch.norm(embeddings_coil_noise.weight.data - self.embeddings_coil.weight.data)
            # noise_vol = torch.norm(embeddings_vol_noise.weight.data - self.embeddings_vol.weight.data)
            # print(f'Mean added noise to coil embeddings: {noise_coil}, \nMean added noise to volume embeddings: {noise_vol}')
            
            latent_vol = self.embeddings_vol(vol_ids)
            latent_coil = self.embeddings_coil(self.start_idx[vol_ids] + coil_ids)
            self.optimizer.zero_grad(set_to_none=True)
            
            # ADD NOISE TO MODEL
            # model_temp = copy.deepcopy(self.model).to(self.device)
            # model_noise = self.add_noise_model(model_temp)
            # noise_added = self.check_amount_noise(model_noise)
            # print(f'Added noise to model: {noise_added}')
            
            outputs = self.model(coords, latent_vol, latent_coil)
            
            # Can be thought as a moving average (with "stride" `batch_size`) of the loss.
            batch_loss = self.loss_fn(outputs, targets, latent_vol, latent_coil)
            batch_loss.backward()
            self.optimizer.step()
        
            avg_loss += batch_loss.item() * len(inputs)
            n_obs += len(inputs)

        self.scheduler.step()
        avg_loss = avg_loss / n_obs
        
            
        return avg_loss

    ###########################################################################
    ###########################################################################
    ###########################################################################

    @torch.no_grad()
    def predict(self, vol_id, shape, left_idx, right_idx, center_vals):
        """Reconstruct MRI volume (k-space)."""
        self.model.eval()
        n_slices, n_coils, height, width = shape

        # Create tensors of indices for each dimension
        kx_ids = torch.cat([torch.arange(left_idx), torch.arange(right_idx, width)])
        ky_ids = torch.arange(height)
        kz_ids = torch.arange(n_slices)
        coil_ids = torch.arange(n_coils)

        # Use meshgrid to create expanded grids
        kspace_ids = torch.meshgrid(kx_ids, ky_ids, kz_ids, coil_ids, indexing="ij")
        kspace_ids = torch.stack(kspace_ids, dim=-1).reshape(-1, len(kspace_ids))

        dataset = TensorDataset(kspace_ids)
        dataloader = DataLoader(
            dataset, batch_size=60_000, shuffle=False, num_workers=3
        )
        vol_embeddings = self.embeddings_vol(
            torch.tensor([vol_id] * 60_000, dtype=torch.long, device=self.device)
        )

        volume_kspace = torch.zeros(
            (n_slices, n_coils, height, width, 2),
            dtype=torch.float32,
            device = self.device
        )  

        for point_ids in dataloader:
            point_ids = point_ids[0].to(self.device, dtype=torch.long)
            coords = torch.zeros_like(
                point_ids[:,:-1], dtype=torch.float32, device=self.device
            )
            # Normalize the necessary coordinates for hash encoding to work
            coords[:, 0] = (2 * point_ids[:, 0]) / (width - 1) - 1
            coords[:, 1] = (2 * point_ids[:, 1]) / (height - 1) - 1
            coords[:, 2] = (2 * point_ids[:, 2]) / (n_slices - 1) - 1

            coil_embeddings = self.embeddings_coil(self.start_idx[vol_id] + point_ids[:, 3])

            # Need to add `:len(coords)` because the last batch has a different size (than 60_000).
            outputs = self.model(coords, vol_embeddings[: len(coords)], coil_embeddings)

            # "Fill in" the unsampled region.            
            volume_kspace[
                point_ids[:, 2], point_ids[:, 3], point_ids[:, 1], point_ids[:, 0]
            ] = outputs


        # # Multiply by the normalization constant.
        volume_kspace_weighted = tensor_to_complex_np(
            volume_kspace.detach().cpu() * self.dataloader.dataset.metadata[vol_id]["norm_cste"])
        volume_kspace = tensor_to_complex_np(volume_kspace.detach().cpu())
        
        coils_img = []
        for i in range(4):
            coils_img.append(np.abs(inverse_fft2_shift(volume_kspace_weighted)[:,i]))
            
        self.model.train()
        
        return volume_kspace, volume_kspace_weighted, coils_img

    @torch.no_grad() 
    def _log_performance(self, epoch_idx):
        for vol_id in self.dataloader.dataset.metadata.keys():
            # Predict volume image.
            shape = self.dataloader.dataset.metadata[vol_id]["shape"]
            center_data = self.dataloader.dataset.metadata[vol_id]["center"]
            left_idx, right_idx, center_vals = (
                center_data["left_idx"],
                center_data["right_idx"],
                center_data["vals"],
            )
            cste_mod = self.dataloader.dataset.metadata[vol_id]["norm_cste"]
            cste_arg = np.pi / 180
            
            ## Fully sampled kspace prediction
            volume_kspace_unweighted, volume_kspace, coils_img = self.predict(vol_id, shape, left_idx, right_idx, center_vals)
            y_kspace_data = tensor_to_complex_np(self.kspace_gt[vol_id])
            
            ###### predict the modulus
            mask = self.dataloader.dataset.metadata[vol_id]["mask"].squeeze(-1).expand(shape).numpy()

            ###### predict the edges - image
            y_img_edges = rss(inverse_fft2_shift(volume_kspace))
            
            ###### predict the center - image
            volume_kspace[..., left_idx:right_idx] = center_vals
            y_img_edges_center = rss(inverse_fft2_shift(volume_kspace))
            
            ######## Modulus
            modulus = np.abs(fft2_shift(y_img_edges_center))
            phase = np.angle(fft2_shift(y_img_edges_center))
            modulus[...,left_idx:right_idx] = 0

            for slice_id in range(shape[0]):
                self._plot_2subplots(y_img_edges, 'Model predictions',
                                    y_img_edges_center, 'Prediction + centre', 
                                    slice_id, epoch_idx, 
                                    f"prediction/vol_{vol_id}_slice_{slice_id}/volume_img",
                                    'gray')

                # self._plot_3subplots(np.abs(y_kspace_data_rss/cste_mod), 'M · ydata',
                #                     np.abs(y_kspace_prediction_rss/cste_mod), '(1-M) · ypred', 
                #                     np.abs(y_kspace_final_rss/cste_mod), 'M · ydata + (1-M)·ypred',
                #                     slice_id, epoch_idx, 
                #                     f"prediction/vol_{vol_id}_slice_{slice_id}/kspace composition",
                #                     'viridis')

                # self._plot_2subplots(self.ground_truth[vol_id], 'groundtruth vol',
                #                     raw_img_edges, 'groundtruth edges', 
                #                     slice_id, epoch_idx, 
                #                     f"groundtruth/vol_{vol_id}_slice_{slice_id}", 'gray')
                
                self._plot_2subplots(modulus/cste_mod, 'kspace predicted',
                                    phase/cste_arg, 'kspace predicted + acquired', 
                                    slice_id, epoch_idx, 
                                    f"prediction/vol_{vol_id}_slice_{slice_id}/kspace logarigthm", 'viridis')

            # ############################################################
            ### Comparison metrics for the volume image w center and the groundtruth
            nmse_val = nmse(self.ground_truth[vol_id], y_img_edges_center)
            self.writer.add_scalar(f"eval/vol_{vol_id}/nmse_wcenter", nmse_val, epoch_idx)

            psnr_val = psnr(self.ground_truth[vol_id], y_img_edges_center)
            self.writer.add_scalar(f"eval/vol_{vol_id}/psnr_wcenter", psnr_val, epoch_idx)

            ssim_val = ssim(self.ground_truth[vol_id], y_img_edges_center)
            self.writer.add_scalar(f"eval/vol_{vol_id}/ssim_wcenter", ssim_val, epoch_idx)

            ## Update.
            self.last_nmse[vol_id] = nmse_val
            self.last_psnr[vol_id] = psnr_val
            self.last_ssim[vol_id] = ssim_val


    @torch.no_grad()
    def _plot_3subplots(
        self, data_1, title1, data_2, title2, data_3, title3, slice_id, epoch_idx, tag, map
        ):
        fig = plt.figure(figsize=(20,30))
        plt.subplot(1,3,1)
        plt.imshow(data_1[slice_id], cmap=map)
        plt.title(title1)
        plt.axis('off')
        
        plt.subplot(1,3,2)
        plt.imshow(data_2[slice_id], cmap=map)
        plt.title(title2)
        plt.axis('off')
        
        plt.subplot(1,3,3)                
        plt.imshow(data_3[slice_id], cmap=map)
        plt.title(title3)
        plt.axis('off')
        
            
        self.writer.add_figure(
            tag,
            fig,
            global_step=epoch_idx,
        )
        plt.close(fig)
        
    @torch.no_grad()
    def _plot_2subplots(
        self, data_1, title1, data_2, title2, slice_id, epoch_idx, tag, map
        ):
        fig = plt.figure(figsize=(20,20))
        plt.subplot(1,2,1)
        plt.imshow(data_1[slice_id], cmap=map)
        plt.title(title1)
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(data_2[slice_id], cmap=map)
        plt.title(title2)
        plt.axis('off')
            
        self.writer.add_figure(
            tag,
            fig,
            global_step=epoch_idx,
        )
        plt.close(fig)
    
    @torch.no_grad()
    def _log_coil_embeddings(
        self, epoch_idx, tag
        
    ):
        full_coil_embeddings, vol_embeddings = self.embeddings_coil.weight.data.cpu(), self.embeddings_vol.weight.data.cpu()
        nvols = vol_embeddings.shape[0]
        fig = plt.figure(figsize=(nvols*6,12))

        for i in range(nvols):
            if i != nvols-1:
                end_coil = self.start_idx[i+1]
                coil_embeddings = full_coil_embeddings[self.start_idx[i] : end_coil]
            else:
                coil_embeddings = full_coil_embeddings[self.start_idx[i]:]
        
            plt.subplot(nvols,1,i+1)
            plt.title(f"Vol: {i+1}")
            plt.hist(coil_embeddings.flatten(), bins = 100)
            if i != nvols - 1:
                plt.xticks([])
        plt.suptitle('Coils', fontsize=30)
        plt.tight_layout()
        self.writer.add_figure(tag, fig, global_step=epoch_idx)
        plt.close(fig)
        
    @torch.no_grad()
    def _log_weight_info(self, epoch_idx):
        """Log weight values and gradients."""
        for module, case in zip(
            [self.model, self.embeddings_vol, self.embeddings_coil], ["network", "embeddings_vol", "embeddings_coil"]
        ):
            for name, param in module.named_parameters():
                subplot_count = 1 if param.data is None else 2
                fig = plt.figure(figsize=(8 * subplot_count, 5))

                plt.subplot(1, subplot_count, 1)
                plt.hist(param.data.cpu().numpy().flatten(), bins=100, log=True)
                # plt.hist(param.data.cpu().numpy().flatten(), bins='auto', log=True)
                plt.title("Values")

                if param.grad is not None:
                    plt.subplot(1, subplot_count, 2)
                    # plt.hist(param.grad.cpu().numpy().flatten(), bins='auto', log=True)
                    plt.hist(param.grad.cpu().numpy().flatten(), bins=100, log=True)
                    plt.title("Gradients")

                tag = name.replace(".", "/")
                self.writer.add_figure(
                    f"params/{case}/{tag}", fig, global_step=epoch_idx
                )
                plt.close(fig)

    @torch.no_grad()
    def _save_checkpoint(self, epoch_idx):
        """Save current state of the training process."""
        # Ensure the path exists.
        path = self.path_to_out / self.timestamp / "checkpoints"
        os.makedirs(path, exist_ok=True)

        path_to_file = path / f"epoch_{epoch_idx:04d}.pt"

        # Prepare state to save.
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "embedding_vol_state_dict": self.embeddings_vol.state_dict(),
            "embedding_coil_state_dict": self.embeddings_coil.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        # Save trainer state.
        torch.save(save_dict, path_to_file)

    @torch.no_grad()
    def _log_information(self, loss):
        """Log 'scientific' and 'nuissance' hyperparameters."""

        hparam_metrics = {"hparam/loss": loss}
        hparam_metrics["hparam/eval_metric/psnr"] = np.mean(self.last_psnr)
        # hparam_metrics["hparam/eval_metric/psnr_stdev"] = np.std(self.last_psnr)

        hparam_metrics["hparam/eval_metric/ssim"] = np.mean(self.last_ssim)
        # hparam_metrics["hparam/eval_metric/ssim_stdev"] = np.std(self.last_ssim)
        
        hparam_metrics["hparam/eval_metric/nmse"] = np.mean(self.last_nmse)
        # hparam_metrics["hparam/eval_metric/nmse_stdev"] = np.std(self.last_nmse)   
        self.writer.add_hparams(self.hparam_info, hparam_metrics)


###########################################################################
###########################################################################
###########################################################################


##################################################
# Loss Functions
##################################################


class MAELoss:
    """Mean Absolute Error Loss Function."""

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, predictions, targets):
        loss = torch.sum(torch.abs(predictions - targets), dim=-1)
        return self.gamma * torch.mean(loss)


class DMAELoss:
    """Dynamic Mean Absolute Error Loss Function."""

    def __init__(self, gamma=100):
        self.gamma = gamma

    def __call__(self, predictions, targets, epoch_id):
        loss = torch.sum(torch.abs(predictions - targets), dim=-1)
        return (epoch_id / self.gamma + 1) * torch.mean(loss)


class MSELoss:
    """Mean Squared Error Loss Function."""

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, predictions, targets):
        predictions = torch.view_as_complex(predictions)
        targets = torch.view_as_complex(targets)

        loss = ((predictions - targets).abs()) ** 2

        return self.gamma * torch.mean(loss)


class MSEL2Loss:
    """Mean Squared Error Loss Function with L2 (latent embedding) Regularization."""

    def __init__(self, sigma, gamma):
        self.gamma = gamma

    def __call__(self, predictions, targets, embeddings_vol, embeddings_coil):
        predictions = torch.view_as_complex(predictions)
        targets = torch.view_as_complex(targets)

        loss = ((predictions - targets).abs()) ** 2

        reg_coil = (embeddings_vol**2).sum(axis=-1)
        reg_vol = (embeddings_coil**2).sum(axis=-1)

        return torch.mean(loss) + self.gamma * (torch.mean(reg_coil) + torch.mean(reg_vol)) * 0.5


class MSEDistLoss:
    """Mean Squared Error Loss Function."""

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, predictions, targets, kcoords):
        predictions = torch.view_as_complex(predictions)
        targets = torch.view_as_complex(targets)

        distance_to_center = torch.sqrt(kcoords[:, 0] ** 2 + kcoords[:, 1] ** 2)

        loss = (1 + 1 / distance_to_center) * ((predictions - targets).abs()) ** 2

        return self.gamma * torch.mean(loss)


class HDRLoss:
    def __init__(self, sigma, epsilon, factor):
        self.sigma = sigma
        self.epsilon = epsilon
        self.factor = factor

    def __call__(self, predictions, targets, kcoords):
        predictions = torch.view_as_complex(predictions)
        targets = torch.view_as_complex(targets)

        loss = ((predictions - targets).abs() / (targets.abs() + self.epsilon)) ** 2

        if self.factor > 0.0:
            dist_to_center2 = kcoords[:, 0] ** 2 + kcoords[:, 1] ** 2
            filter_value = torch.exp(-dist_to_center2 / (2 * self.sigma**2))

            reg_error = predictions - predictions * filter_value
            reg = self.factor * (reg_error.abs() / (targets.abs() + self.epsilon)) ** 2

            return loss.mean() + reg.mean()

        else:
            return loss.mean()


class LogL2Loss(torch.nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        log_y_pred = torch.log(fastmri.complex_abs(y_pred) + self.epsilon)
        log_y_true = torch.log(fastmri.complex_abs(y_true) + self.epsilon)
        loss = torch.mean((log_y_pred - log_y_true) ** 2)
        return loss


###########################################################################
###########################################################################
###########################################################################

##################################################
# Evaluation Metrics
##################################################


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array(0.0)
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]