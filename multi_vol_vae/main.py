import time
from itertools import chain
import torch
from config.config_utils import (
    handle_reproducibility,
    load_config,
    parse_args,
    save_config,
)

from datasets import KCoordDataset


from torch.utils.data import DataLoader
from train_utils import *
from models.d2c_vae.autoencoder_unet import Autoencoder
from models.d2c_vae.mlp import MLP
import torchvision.datasets as dsets
from torchvision import transforms
from PIL import Image 


def main():
    args = parse_args()
    config = load_config(args.config)
    config["device"] = args.device

    rs_numpy, rs_torch = handle_reproducibility(config["seed"])
    print(f'Connected to GPU: {torch.cuda.is_available()}')
    torch.set_default_dtype(torch.float32)

    # transform_list = transforms.Compose([
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ])
        
    loader_config = config["dataloader"]
    dataset_config = config["dataset"]
    
    latent_dim = config['ddconfig']['out_ch']
    
    model = MLP(**config["mlpconfig"])
    vaemodel = Autoencoder(ddconfig=config["ddconfig"], 
                        embed_dim=latent_dim)

    stage = config["runtype"]
    

    
    if stage == "test":
        assert (
            "model_checkpoint" in config.keys()
        ), "Error: Trying to start a test run without a model checkpoint."
        
        # prepare validation dataset
        test_data = dsets.ImageFolder(dataset_config["path_to_test_data"], transform=transform_list)
        masked_data = mask_data(test_data, stage,
                with_center = dataset_config["with_center"], 
                acceleration = dataset_config["acc"], 
                center_frac = dataset_config["center_frac"],
                )
        mask = masked_data.get_mask()
        val_loader = torch.utils.data.DataLoader(masked_data, 
                                        batch_size=loader_config["batch_size"],
                                        shuffle=False,
                                        num_workers=2,
                                        pin_memory=loader_config["pin_memory"],
                                        drop_last=True)
        
        # Load checkpoint.
        model_state_dict = torch.load(config["model_checkpoint"])["model_state_dict"]
        vae_state_dict = torch.load(config["model_checkpoint"])["vae_state_dict"]
        model.load_state_dict(model_state_dict)
        vaemodel.load_state_dict(vae_state_dict)

        print("Checkpoint loaded successfully.")

    
        # Freeze the MLP model and Encoder weights
        for _, tensor in model.named_parameters():
            tensor.requires_grad = False
        
        # for _, tensor in vaemodel.encoder.named_parameters():
        #     tensor.requires_grad = False
        for _, tensor in vaemodel.named_parameters():
            tensor.requires_grad = False

        dataloader, dataset = val_loader, test_data
        
    elif stage == "train":
        # prepare dataset
        
        dataset = KCoordDataset(
                path_to_data=config["dataset"]["path_to_data"],
                n_volumes=config["dataset"]["n_volumes"],
                n_slices=config["dataset"]["n_slices"],
                acceleration=config["dataset"]["acceleration"],
                center_frac=config["dataset"]["center_frac"],
                mode=config["runtype"],
                )
        
        train_loader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=loader_config["batch_size"],
                                        shuffle=True,
                                        num_workers=2,
                                        pin_memory=loader_config["pin_memory"],
                                        drop_last=True)
        
        val_loader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=loader_config["batch_size"],
                                        shuffle=False,
                                        num_workers=2,
                                        pin_memory=loader_config["pin_memory"],
                                        drop_last=True)
        mask = dataset.get_mask()
        if "model_checkpoint" in config.keys():
            model_state_dict = torch.load(config["model_checkpoint"])[
                "model_state_dict"
            ]
            vae_state_dict = torch.load(config["model_checkpoint"])[
                "vae_state_dict"
            ]
            model.load_state_dict(model_state_dict)
            vaemodel.load_state_dict(vae_state_dict)
            
            print("Checkpoint loaded successfully.")


        dataloader, dataset = train_loader, dataset
        trainer = Trainer(
            dataset = dataset,
            train_dataloader=dataloader,
            val_loader = val_loader,
            mask = mask,
            model = model,
            vae = vaemodel,
            stage = stage,
            config=config,
        )


    else:
        raise ValueError("Incorrect runtype (must be `train` or `test`).")


    print(f"model {model}")
    print(f"Autoencode {vaemodel}")
    print(config)
    print(f"Number of steps per epoch: {len(dataloader)}")

    print(f"Starting {stage} process...")
    t0 = time.time()
    
    trainer.train()

    save_config(config)

    t1 = time.time()
    print(f"Time it took to run: {(t1-t0)/60} min")


if __name__ == "__main__":
    main()