# [DDRM-PR: Fourier Phase Retrieval using Denoising Diffusion Restoration Models](https://arxiv.org/abs/2501.03030)
Mehmet Onurcan Kaya and Figen S. Oktem

This repository contains the official codes for the paper "[DDRM-PR: Fourier Phase Retrieval using Denoising Diffusion Restoration Models](https://arxiv.org/abs/2501.03030)", to appear in Applied Optics. ([arXiv](https://arxiv.org/abs/2501.03030))

## Abstract
Diffusion models have demonstrated their utility as learned priors for solving various inverse problems. However, most existing approaches are limited to linear inverse problems. This paper exploits the efficient and unsupervised posterior sampling framework of Denoising Diffusion Restoration Models (DDRM) for the solution of nonlinear phase retrieval problem, which requires reconstructing an image from its noisy intensity-only measurements such as Fourier intensity. The approach combines the model-based alternating-projection methods with the DDRM to utilize pretrained unconditional diffusion priors for phase retrieval. The performance is demonstrated through both simulations and experimental data. Results demonstrate the potential of this approach for improving the alternating-projection methods as well as its limitations.

## Getting Started
To repeat the experiments, create a conda environment with environment.yml. Then, you can run main.py like in example_run.sh.

To repeat the experiments, pre-trained models should be downloaded from 
[https://drive.google.com/drive/folders/1FK056aWoTSjMJQ6ExjDnj_WlhUI5IUIW?usp=sharing](https://drive.google.com/drive/folders/1FK056aWoTSjMJQ6ExjDnj_WlhUI5IUIW?usp=sharing)

Download the "exp" folder, and, move it to this repo.
The experiments in the paper are also given under "exp/image_samples" folder.

## Citation
Please cite the following paper when using this code or data:
```
@misc{kaya2025ddrmprfourierphaseretrieval,
      title={DDRM-PR: Fourier Phase Retrieval using Denoising Diffusion Restoration Models}, 
      author={Mehmet Onurcan Kaya and Figen S. Oktem},
      year={2025},
      eprint={2501.03030},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2501.03030}, 
}  
```

## Contact
If you have any questions or need help, please feel free to contact me via monka@dtu.dk.

## Acknowledgments
The codebase is largely derived from the official DDRM repository. For further questions or details, you can refer to the original repository: [https://github.com/bahjat-kawar/ddrm](https://github.com/bahjat-kawar/ddrm) 
