import os
from pathlib import Path
from typing import Optional

import fastmri
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils import *
from fastmri.data.transforms import tensor_to_complex_np, to_tensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, TensorDataset
from pisco import *
from utils import *
from torch.utils.tensorboard import SummaryWriter # To print to tensorboard


class Trainer:
    def __init__(
        self, dataloader_consistency, dataloader_pisco, model, loss_fn, optimizer, scheduler, config
    ) -> None:
        self.device = torch.device(config["device"])
        self.n_epochs = config["n_epochs"]

        self.dataloader_consistency = dataloader_consistency
        self.dataloader_pisco = dataloader_pisco
        
        self.model = model.to(self.device)
        
        # If stateful loss function, move its "parameters" to `device`.
        if hasattr(loss_fn, "to"):
            self.loss_fn = loss_fn.to(self.device)
        else:
            self.loss_fn = loss_fn
            
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.log_interval = config["log_interval"]
        self.checkpoint_interval = config["n_epochs"] # Initialize the checkpoint interval as this value
        self.path_to_out = Path(config["path_to_outputs"])
        self.timestamp = config["timestamp"]
        self.add_pisco = config["l_pisco"]["addpisco"]
        self.E_epoch = config["l_pisco"]["E_epoch"]
        self.alpha = config["l_pisco"]["alpha"]
        self.factor = config["l_pisco"]["factor"]
        self.minibatch = config["l_pisco"]["minibatch"]
        self.patch_size = config["l_pisco"]["patch_size"]
        
        self.writer = SummaryWriter(self.path_to_out / self.timestamp)

        # Ground truth (used to compute the evaluation metrics).
        file = self.dataloader_consistency.dataset.metadata[0]["file"]
        with h5py.File(file, "r") as hf:
            self.ground_truth = hf["reconstruction_rss"][()][
                : config["dataset"]["n_slices"]
            ]
            self.kspace_gt = to_tensor(preprocess_kspace(hf["kspace"][()][: config["dataset"]["n_slices"]]))

        # Scientific and nuissance hyperparameters.
        self.hparam_info = config["hparam_info"]
        self.hparam_info["n_layer"] = config["model"]["params"]["n_layers"]
        self.hparam_info["hidden_dim"] = config["model"]["params"]["hidden_dim"]
        # self.hparam_info["resolution_levels"] = config["model"]["params"]["levels"]
        self.hparam_info["batch_size"] = config["dataloader"]["batch_size"]
        # self.hparam_info["pisco_weightfactor"] = config["l_pisco"]["factor"]
        
        print(self.hparam_info)

        # Evaluation metrics for the last log.
        self.last_nmse = [0] * len(self.dataloader_consistency.dataset.metadata)
        self.last_psnr = [0] * len(self.dataloader_consistency.dataset.metadata)
        self.last_ssim = [0] * len(self.dataloader_consistency.dataset.metadata)

    ###########################################################################
    ###########################################################################
    ###########################################################################

    def train(self):
        """Train the model across multiple epochs and log the performance."""
        empirical_risk = 0
        empirical_pisco = 0
        for epoch_idx in range(self.n_epochs):
            empirical_risk = self._train_one_epoch()
            
            print(f"EPOCH {epoch_idx}    avg loss: {empirical_risk}\n")
            
            ### If there is Pisco regularization
            if self.add_pisco and (epoch_idx + 1) >= self.E_epoch:
                    empirical_pisco, epoch_res1, epoch_res2, self.batch_grappas = self._train_with_Lpisco()
                    print(f"EPOCH {epoch_idx}  Pisco loss: {empirical_pisco}\n")
                    self.writer.add_scalar("Residuals/Linear", epoch_res1, epoch_idx)
                    self.writer.add_scalar("Residuals/Regularizer", epoch_res2, epoch_idx)
                        
            # Log the errors
            self.writer.add_scalar("Loss/train", empirical_risk, epoch_idx)
            self.writer.add_scalar("Loss/Pisco", empirical_pisco, epoch_idx)
            
            # Log the average residuals
            # TODO: UNCOMMENT WHEN USING LR SCHEDULER.
            # self.writer.add_scalar("Learning Rate", self.scheduler.get_last_lr()[0], epoch_idx)

            if (epoch_idx + 1) % self.log_interval == 0:
                self._log_performance(epoch_idx)
                self._log_weight_info(epoch_idx)
            
            if (epoch_idx + 1) % self.checkpoint_interval == 0:
                # Takes ~3 seconds.
                self._save_checkpoint(epoch_idx)
                

        self._log_information(empirical_risk)
        self.writer.close()
        
    def _train_one_epoch(self):
        # Also known as "empirical risk".
        avg_loss = 0.0
        n_obs = 0

        self.model.train()
        
        
        for inputs, targets in self.dataloader_consistency:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            
            # Can be thought as a moving average (with "stride" `batch_size`) of the loss.
            batch_loss = self.loss_fn(outputs, targets)
            # NOTE: Uncomment for some of the loss functions (e.g. 'MSEDistLoss').
            # batch_loss = self.loss_fn(outputs, targets, inputs)

            batch_loss.backward()
            self.optimizer.step()

            avg_loss += batch_loss.item() * len(inputs)
            n_obs += len(inputs)

        avg_loss = avg_loss / n_obs
        return avg_loss
    
    
    def _train_with_Lpisco (self):
        self.model.train()
        vol_id = 0
        n_obs = 0
        shape = self.dataloader_pisco.dataset.metadata[vol_id]["shape"]
        _, n_coils, _, _ = shape
        
        batch_grappa = []
        err_pisco = 0
        res1 = 0
        res2 = 0
        
        for inputs, _ in self.dataloader_pisco:
            
            #### Compute grid 
            t_coordinates, patch_coordinates, Nn = get_grappa_matrixes(inputs, shape, patch_size=self.patch_size, normalized=True)
            
            # Estimate the minibatch list of Ws together with the averaged residuals of the minibatch 
            ws, ws_nograd, batch_r1, batch_r2 = self.predict_ws(t_coordinates, patch_coordinates, n_coils, Nn)
                        
            # Compute the pisco loss
            batch_Lp = L_pisco(ws) * self.factor
            
            print(f'Batch pisco loss: {batch_Lp}')
            assert batch_Lp.requires_grad, "batch_Lp does not require gradients."
        
            # Update the model based on the Lpisco loss
            batch_Lp.backward()
        
            self.optimizer.step()

            # Compute the mean_batch_grappa_matrix
            w_grappa = np.mean(ws_nograd, axis = 0)
            batch_grappa.append(w_grappa)

            err_pisco += batch_Lp.item()
            res1 += batch_r1
            res2 += batch_r2
            n_obs += 1
            
        return err_pisco/n_obs, res1/n_obs, res2/n_obs, batch_grappa

    ##########################################################################
    ##########################################################################
    ##########################################################################
    
    def predict_ws (self, t_coordinates, patch_coordinates, n_coils, Nn):
        
        t_predicted = torch.zeros((t_coordinates.shape[0], n_coils), dtype=torch.complex64)
        t_coordinates, patch_coordinates = t_coordinates.to(self.device), patch_coordinates.to(self.device)
        
        # nxn Neighbourhood patch surrounding the target point
        neighborhood_corners = torch.zeros((t_coordinates.shape[0], Nn, n_coils), dtype=torch.complex64)

        for idx in range(n_coils):
            # Predict value of t 
            t_predicted[:, idx] = torch.view_as_complex(self.model(t_coordinates[:, idx, :]))
            for nn in range(Nn):
                # Predict value of neighbors
                neighborhood_corners[:, nn, idx] = torch.view_as_complex(self.model(patch_coordinates[:, nn, idx, :])).detach()
        
        # ##### Estimate the Ws for a random subset of values
        T_s, P_s = split_matrices_randomly(t_predicted, neighborhood_corners, self.minibatch)
        Ns = len(P_s)
        
        # Estimate the Weight matrixes
        Ws = []
        Ws_nograd = []
        elem1 = 0
        elem2 = 0
        
        
        for i, t_s in enumerate(T_s):
            p_s = P_s[i].flatten(1)
            
            # ws, res1, res2 = compute_Lsquares(p_s, t_s, self.alpha)
            ws = compute_Lsquares(p_s, t_s, self.alpha)
            Ws.append(ws)
            
            ws_nograd = ws.detach()
            Ws_nograd.append(ws_nograd)

        return Ws, Ws_nograd, elem1/Ns, elem2/Ns
    
    @torch.no_grad()
    def predict(self, vol_id, shape, left_idx, right_idx, center_vals, epoch_idx):
        """Reconstruct MRI volume (k-space)."""
        self.model.eval()
        n_slices, n_coils, height, width = shape
        norm_cte = [width, height, n_slices, n_coils]
        
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
            dataset, batch_size=60_000, shuffle=False, num_workers=2
        )

        volume_kspace = torch.zeros(
            (n_slices, n_coils, height, width, 2),
            device=self.device,
            dtype=torch.float32,
        )
        
        for point_ids in dataloader:
            point_ids = point_ids[0].to(self.device, dtype=torch.long)
            coords = torch.zeros_like(
                point_ids, dtype=torch.float32, device=self.device
            )
            
            coords[:, 0] = (2 * point_ids[:, 0]) / (width - 1) - 1
            coords[:, 1] = (2 * point_ids[:, 1]) / (height - 1) - 1
            coords[:, 2] = (2 * point_ids[:, 2]) / (n_slices - 1) - 1
            coords[:, 3] = (2 * point_ids[:, 3]) / (n_coils - 1) - 1
            
            outputs = self.model(coords)
            # "Fill in" the unsampled region.
            volume_kspace[
                point_ids[:, 2], point_ids[:, 3], point_ids[:, 1], point_ids[:, 0]
            ] = outputs
            
    
        # Multiply by the normalization constant.
        volume_kspace = (
            volume_kspace * self.dataloader_consistency.dataset.metadata[vol_id]["norm_cste"]
        )

        volume_kspace = tensor_to_complex_np(volume_kspace.detach().cpu())
        

        if self.add_pisco and (epoch_idx + 1) >= self.E_epoch:
            grappa_volume = torch.zeros(shape, dtype = torch.complex64)
            # If it is a checkpoint, recalculate the grappa volume, as the mean of the list of grappa matrixes 
            w_grappa = torch.tensor(np.mean(self.batch_grappas, axis=0)) # Size: Nn·Nc x Nc
                
            # Now predict the sensitivities (accuracy of the Ws grappa matrixes)
            for points_ids in dataloader:
                points_ids = points_ids[0]
                t_coors, nn_coors, Nn = get_grappa_matrixes(points_ids, shape, patch_size=self.patch_size, normalized=False)
                
                den_t_coors = torch.zeros((t_coors.shape), dtype=torch.int)
                den_nn_coors = torch.zeros((nn_coors.shape), dtype=torch.int)
                for idx in range(len(shape)):
                    den_t_coors[...,idx] = denormalize_fn(t_coors[...,idx], norm_cte[idx])
                    den_nn_coors[...,idx] = denormalize_fn(nn_coors[...,idx], norm_cte[idx])

                nn_kspacevals = torch.tensor(tensor_to_complex_np(self.kspace_gt[den_nn_coors[...,2], den_nn_coors[...,3], den_nn_coors[...,1], den_nn_coors[...,0]]),
                                        dtype = torch.complex64)
                ps_kspacevals = nn_kspacevals.view(t_coors.shape[0], Nn*n_coils)
                t_kspacevals = torch.matmul(ps_kspacevals, w_grappa)  # NOTE : Computed value based on neighbouring patch of 3x3 and estimated grappa mean
                grappa_volume[den_t_coors[...,2], den_t_coors[...,3], den_t_coors[...,1], den_t_coors[...,0]] = t_kspacevals
            
            grappa_volume = np.abs(inverse_fft2_shift(grappa_volume))
        else:
            grappa_volume = []
            
    
        self.model.train()
        return volume_kspace, grappa_volume

    ###########################################################################
    ###########################################################################
    ###########################################################################
    @torch.no_grad() 
    def _log_performance(self, epoch_idx, vol_id = 0):
            # Predict volume image.
            
            shape = self.dataloader_consistency.dataset.metadata[vol_id]["shape"]
            center_data = self.dataloader_consistency.dataset.metadata[vol_id]["center"]
            left_idx, right_idx, center_vals = (
                center_data["left_idx"],
                center_data["right_idx"],
                center_data["vals"],
            )

            volume_kspace, grappa_volume = self.predict(vol_id, shape, left_idx, right_idx, center_vals, epoch_idx)
            # cste_mod = self.dataloader_consistency.dataset.metadata[vol_id]["norm_cste"]
            
            y_kspace_data = tensor_to_complex_np(self.kspace_gt)
            mask = self.dataloader_consistency.dataset.metadata[vol_id]["mask"].squeeze(-1).expand(shape).numpy()
            
            # breakpoint()
            y_kspace_data_u = y_kspace_data * (mask)
            y_kspace_prediction_u = volume_kspace * (1-mask)
            y_kspace_final = y_kspace_data_u + y_kspace_prediction_u
            
            ###### predict the edges - image
            img_predicted = rss(inverse_fft2_shift(volume_kspace))
            
            ###### predict the center - image
            img_consistency = rss(inverse_fft2_shift(y_kspace_final))
            
            modulus_kspace = np.abs(fft2_shift(img_predicted))
            phase_kspace = np.angle(fft2_shift(img_predicted))
            cste_arg = np.pi/180


            for slice_id in range(shape[0]):

                self._plot_2subplots(img_predicted, 'pred',
                                    img_consistency, 'pred + centre', 
                                    slice_id, epoch_idx, 
                                    f"prediction/vol_{vol_id}_slice_{slice_id}/volume_img", 'gray')
                
                self._plot_2subplots(modulus_kspace, 'Modulus kspace predict',
                                    phase_kspace/cste_arg, 'Phase kspace', 
                                    slice_id, epoch_idx, 
                                    f"kspace/vol_{vol_id}_slice_{slice_id}", 'viridis')
                
                if self.add_pisco and (epoch_idx + 1) >= self.E_epoch:
                    fig = plt.figure(figsize=(10,5))
                    for i in range(4): # Plot the first 4 coils
                        plt.subplot(1,4,i+1)
                        plt.imshow(grappa_volume[slice_id,i,...], cmap='gray')
                        plt.axis('off')

                    self.writer.add_figure(
                        f"prediction/vol_{vol_id}/slice{slice_id}/coils_img",
                        fig,
                        global_step=epoch_idx,
                    )
                    plt.close(fig)
                
                
            # ############################################################
            # # # Comparison metrics for the volume image w center and the groundtruth
            nmse_val = nmse(self.ground_truth, img_predicted)
            self.writer.add_scalar(f"eval/vol_{vol_id}/nmse_wcenter", nmse_val, epoch_idx)

            psnr_val = psnr(self.ground_truth, img_predicted)
            self.writer.add_scalar(f"eval/vol_{vol_id}/psnr_wcenter", psnr_val, epoch_idx)

            ssim_val = ssim(self.ground_truth, img_predicted)
            self.writer.add_scalar(f"eval/vol_{vol_id}/ssim_wcenter", ssim_val, epoch_idx)
            
            self.last_nmse = nmse_val
            self.last_psnr = psnr_val
            self.last_ssim = ssim_val

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
        # plt.colorbar()
        
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
    def _log_weight_info(self, epoch_idx):
        """Log weight values and gradients."""
        for name, param in self.model.named_parameters():
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
            self.writer.add_figure(f"params/{tag}", fig, global_step=epoch_idx)
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
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        # Save trainer state.
        torch.save(save_dict, path_to_file)

    @torch.no_grad()
    def _log_information(self, loss):
        """Log 'scientific' and 'nuissance' hyperparameters."""

        # if hasattr(self.model, "activation"):
        #     self.hparam_info["hidden_activation"] = type(self.model.activation).__name__
        # elif type(self.model).__name__ == "Siren":
        #     self.hparam_info["hidden_activation"] = "Sine"
        # if hasattr(self.model, "out_activation"):
        #     self.hparam_info["output_activation"] = type(
        #         self.model.out_activation
        #     ).__name__
        # else:
        #     self.hparam_info["output_activation"] = "None"

        hparam_metrics = {"hparam/loss": loss}
        hparam_metrics["hparam/eval_metric/nmse"] = np.mean(self.last_nmse)
        hparam_metrics["hparam/eval_metric/psnr"] = np.mean(self.last_psnr)
        hparam_metrics["hparam/eval_metric/ssim"] = np.mean(self.last_ssim)
        
        self.writer.add_hparams(self.hparam_info, hparam_metrics)

        # Log model's architecture.
        inputs, _ = next(iter(self.dataloader_consistency))
        inputs = inputs.to(self.device)
        self.writer.add_graph(self.model, inputs)


###########################################################################
###########################################################################
##########################################################################