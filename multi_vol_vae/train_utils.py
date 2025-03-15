import os
from pathlib import Path
from typing import Optional
from itertools import chain
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils import *
from torchvision import utils as vtils
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from lpips import LPIPS
from fastmri.data.transforms import tensor_to_complex_np, to_tensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.tensorboard import SummaryWriter
from general_utils import multiscale_image_transform

OPTIMIZER_CLASSES = {
    "Adam": Adam,
    "AdamW": AdamW,
    "SGD": SGD,
}

SCHEDULER_CLASSES = {"StepLR": StepLR}

class Trainer:
    def __init__(
        self, dataset, train_dataloader, val_loader, mask, model, vae, stage, config
        ) -> None:
        self.device = torch.device(config["device"])
        self.n_epochs = config["n_epochs"]
        self.dataloader = train_dataloader
        self.val_loader = val_loader
        self.model, self.vae = model.to(self.device), vae.to(self.device)
        
        self.kl_coeff = config["lossconfig"]["kl_max_coeff"]
        self.log_interval = config["log_interval"]
        self.checkpoint_interval = config["checkpoint_interval"]
        self.path_to_out = Path(config["path_to_outputs"])
        self.timestamp = config["timestamp"]
        
        self.results_pth = os.path.join(config['results_folder'], self.timestamp)
        self.stage = stage
        self.mask = mask.to(self.device).bool()
        os.makedirs(self.results_pth, exist_ok=True)
        

        self.config = config
        # self.writer = SummaryWriter(self.path_to_out / self.timestamp)
        self.perceptual_loss = LPIPS().eval().to(self.device)
        # Ground truth (used to compute the evaluation metrics).
        self.ground_truth = []
        self.kspace_gt = []

        # Evaluation metrics for the last log.
        self.last_nmse = [0] * len(dataset)
        self.last_psnr = [0] * len(dataset)
        self.last_ssim = [0] * len(dataset)

    ###########################################################################
    ###########################################################################
    ###########################################################################

    def train(self):
        """Train the model across multiple epochs and log the performance."""
        empirical_risk = 0
        if self.stage == 'train':
            self.optimizer = OPTIMIZER_CLASSES[self.config["optimizer"]["id"]](
                chain(self.model.parameters(), self.vae.parameters()),
                **self.config["optimizer"]["params"],
            )
    
            self.scheduler = SCHEDULER_CLASSES[self.config["scheduler"]["id"]](
                self.optimizer, **self.config["scheduler"]["params"]
            )
            
            for epoch_idx in range(self.n_epochs):
                empirical_avgloss, empirical_recon, empirical_kld = self._train_one_epoch(epoch_idx)

                print(f"EPOCH {epoch_idx}    avg loss: {empirical_avgloss}\n avg recon: {empirical_recon} \n avg_kld: {empirical_kld}")
                # print(f'KL divergence: {empirical_kld}, Recon. loss: {empirical_recon}')

                if (epoch_idx + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(epoch_idx)
                
                # Visualization of the training process
                if epoch_idx % self.log_interval == 0:
                    self.model.eval()
                    self.vae.eval()
                    plt.figure(figsize=(20,20))
                    for idx in range(3):
                        targets = self.val_loader.dataset[idx]
                        targets = targets.to(self.device) # magnitude and phase (mean coil magnitude, mean phase magnitude)
                        target, coords, scale, y = multiscale_image_transform(targets, 320, False, self.device)
                        self.optimizer.zero_grad()                
                        # VAE, sample latent
                        
                        posterior = self.vae.encode(targets.unsqueeze(0))
                        pe = self.vae.decode(posterior.sample())
                        # Train MLP
                        output = self.model(coords, pe, si=scale) #dim: bsz, 2, 320, 320
                        
                        pred_ks_n = output.detach().cpu()
                        pred_ks_n = torch.view_as_complex(pred_ks_n.squeeze(0).permute(1,2,0).contiguous())
                        
                        quant = self.val_loader.dataset.metadata[idx]["plot_cste"]
                        
                        mod = torch.exp(-pred_ks_n.abs() * quant)
                        phase = pred_ks_n.angle()

                        pred_ks = mod * torch.exp(1j*phase)
                        img_pred = np.abs(inverse_fft2_shift(pred_ks))
                        
                        plt.subplot(2,3,idx+1)
                        plt.imshow(np.abs(pred_ks_n))
                        plt.axis('off')
                        
                        plt.subplot(2,3,idx+4)
                        plt.imshow(img_pred, cmap='gray')
                        plt.axis('off')
                        
                    save_path = os.path.join(self.results_pth, f'step_{epoch_idx}.png')
                    plt.savefig(save_path, bbox_inches = 'tight')
                    plt.close()
                    # vtils.save_image(output, save_path, normalize=True, scale_each=True)
                    self.model.train()
                    self.vae.train()
                    
        elif self.stage == 'test':
            self.latents = []
            
            for batch_idx, targets in enumerate(self.dataloader):
                targets = targets.to(self.device) # magnitude and phase (mean coil magnitude, mean phase magnitude)
                target, coords, scale, y = multiscale_image_transform(targets, 320, False, self.device)
                posterior = self.vae.encode(target).sample()

                self.latents.append(posterior.requires_grad_(True))
        
            self.optimizer = OPTIMIZER_CLASSES[self.config["optimizer"]["id"]](
                    self.latents,
                    **self.config["optimizer"]["params"],
                )
            self.scheduler = SCHEDULER_CLASSES[self.config["scheduler"]["id"]](
                self.optimizer, **self.config["scheduler"]["params"]
            )

            for epoch_idx in range(self.n_epochs-1):
                empirical_recon = self._val_one_epoch(epoch_idx)
                
                print(f"EPOCH {epoch_idx}    Recon loss: {empirical_recon}\n")
                
                if (epoch_idx + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(epoch_idx)
    def _val_one_epoch(self, epoch_idx):
        # Also known as "empirical risk".
        n_obs = 0
        self.vae.train()        
        avg_recon = 0.0
        
        for batch_idx, targets in enumerate(self.dataloader):
            targets = targets.to(self.device) # magnitude and phase (mean coil magnitude, mean phase magnitude)
            target, coords, scale, y = multiscale_image_transform(targets, 320, False, self.device)
                
            self.optimizer.zero_grad()   
            posterior = self.latents[batch_idx]
            pe = self.vae.decode(posterior)

            # Train MLP
            output = self.model(coords, pe, si=scale) #dim: bsz, 2, 320, 320
            
            # Losses
            # Reconstruction loss
            self.mask = self.mask.expand(output.shape)            
            masked_output = output * self.mask

            recon_loss = torch.sum(torch.abs(masked_output.contiguous() - target.contiguous()), dim = (1,2,3)).mean()

            recon_loss.backward()
            self.optimizer.step()
            
            avg_recon += recon_loss.item()             
            n_obs += targets.shape[0]
            self.scheduler.step()
            
        avg_recon = avg_recon / n_obs
        
        # Visualization of the training process
        if epoch_idx % self.log_interval == 0:
            save_path = os.path.join(self.results_pth, f'step_{epoch_idx}.png')
            vtils.save_image(output, save_path, normalize=True, scale_each=True)
        
        return avg_recon
        
    def _train_one_epoch(self, epoch_idx):
        # Also known as "empirical risk".
        avg_loss = 0.0
        n_obs = 0
        self.model.train()        
        self.vae.train()
        avg_recon = 0.0
        avg_kld = 0.0

        for batch_idx, targets in enumerate(self.dataloader):

            targets = targets.to(self.device) # magnitude and phase (mean coil magnitude, mean phase magnitude)
            target, coords, scale, y = multiscale_image_transform(targets, 320, False, self.device)
            
            self.optimizer.zero_grad()                

            # VAE, sample latent
            posterior = self.vae.encode(target)
            pe = self.vae.decode(posterior.sample())

            # Train MLP
            output = self.model(coords, pe, si=scale) #dim: bsz, 2, 320, 320
            
            # Losses
            # Reconstruction loss
            recon_loss = torch.sum(torch.abs(output.contiguous() - target.contiguous()), dim = (1,2,3)).mean()
            # # Perceptual loss
            # p_coeff = 1.
            # p_loss = self.perceptual_loss(target.contiguous(), output.contiguous()).mean()
            
            # kl_loss
            kld = posterior.kl(mean=False)
            kld_loss = torch.mean(kld)

            # Combined loss
            total_loss = recon_loss + self.kl_coeff * kld_loss #+ p_coeff * p_loss

            total_loss.backward()
            self.optimizer.step()
            
            avg_recon += recon_loss.item() 
            avg_loss +=  total_loss.item() 
            avg_kld += kld_loss.item() 
            
            n_obs += targets.shape[0]
            self.scheduler.step()
                
        avg_kld = avg_kld / n_obs
        avg_recon = avg_recon / n_obs
        
        return avg_loss, avg_recon , avg_kld

    ###########################################################################
    ###########################################################################
    ###########################################################################

    # @torch.no_grad()
    # def predict(self, target, epoch_idx, img_idx):
    #     """Reconstruct MRI volume (k-space)."""
    #     self.model.eval()
    #     self.vae.eval()
    #     target = target.to(self.device)

    #     # VAE, sample latent
    #     posterior = self.vae.encode(target.unsqueeze(0))
    #     pe = self.vae.decode(posterior.sample())

    #     # Need to add `:len(coords)` because the last batch has a different size (than 60_000).
    #     outputs = self.model(self.grid, pe).cpu()
        
    #     self.model.train()
    #     self.vae.train()
        
    #     save_path = os.path.join(self.results_pth, f'step_{epoch_idx}_img{img_idx}.png')
    #     vtils.save_image(outputs, save_path, normalize=True, scale_each=True)
        
    #     breakpoint()
        
    #     return outputs

    # @torch.no_grad() 
    # def _log_performance(self, epoch_idx):

    #     breakpoint()
    #     self.predict(self.dataset[:5], epoch_idx)

    #         # ############################################################
    #         # ### Comparison metrics for the volume image w center and the groundtruth
    #         # nmse_val = nmse(self.ground_truth[vol_id], y_img_edges)
    #         # # self.writer.add_scalar(f"eval/vol_{vol_id}/nmse_wcenter", nmse_val, epoch_idx)

    #         # psnr_val = psnr(self.ground_truth[vol_id], y_img_edges)
    #         # # self.writer.add_scalar(f"eval/vol_{vol_id}/psnr_wcenter", psnr_val, epoch_idx)

    #         # ssim_val = ssim(self.ground_truth[vol_id], y_img_edges)
    #         # # self.writer.add_scalar(f"eval/vol_{vol_id}/ssim_wcenter", ssim_val, epoch_idx)

    #         # print(f'NMSE : {nmse_val} \nPSRN : {psnr_val}, \nSSIM : {ssim_val}')
    #         ## Update.

    # @torch.no_grad()
    # def singleplane_positional_encoding(self, latent, coords):
    #     return F.grid_sample(latent, coords, padding_mode='border')


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
            "vae_state_dict": self.vae.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        # Save trainer state.
        torch.save(save_dict, path_to_file)


##############################################################################

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
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array(0.0)
    # for slice_num in range(gt.shape[0]):
    ssim = ssim + structural_similarity(
        gt, pred, data_range=maxval
    )

    return ssim 