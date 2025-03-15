# MRI_reconstruction


This code forms part of my research project on Neural Implicit Fields for MRI reconstruction at CVL, ETH Zürich 👩‍💻.

As part of the experiments performed:

- **Single Volume training**

<img src="imgs/single_vol_model.png" alt="My Image" width="500" height="280">

  
Folders relevant 🖇️: `single_vol`, `single_vol_hash`

  **single_vol** contains the implementation of the PISCO regularization method 

  **single_vol_hash** contains the implementation of the hash encodings  

- **Multi Volume training**

<img src="imgs/multivol_model.png" alt="My Image" width="500" height="300">

  
Folders relevant 🖇️: `multi_vol_hash`, `multi_vol_coil`, `multi_gpu_coil`, `multi_gpu_hash`

  **multi_vol_coil** contains the necessary functions to run the multi-volume experiments optimizing the volume and coil embeddings

  **multi_vol_hash** contains the necessary functions to run the multi-volume experiments with the coordinate hash encodings implementation

If a multi-gpu setup is needed, check the adapted codes on the folders: `multi_gpu_coil`, `multi_gpu_hash`
The `config.yaml` file can be found in all the folders, modified accordingly.

Additional experiments inspired by the ideas and code proposed in the paper [DDMI](https://arxiv.org/html/2401.12517#:~:text=To%20address%20this%20limitation%2C%20we%20propose%20D%20omain-agnostic,adaptive%20positional%20embeddings%20instead%20of%20neural%20networks%E2%80%99%20weights) can be found in the folders: `original_vae` and `multi_vol_vae`

 
To setup the environment use the `mri_env.yaml` file containing the relevant packages to be installed 🚀

Check the report `research_project.pdf` to go more in depth on the methodology/ experiments performed 🔎
