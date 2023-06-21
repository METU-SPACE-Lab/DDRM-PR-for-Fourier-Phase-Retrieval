To repeat the experiments, create a conda environment with environment.yml. And, then you can run main.py like this:

```
python main.py --ni --config celeba_pr.yml --doc celeba_dene33 -i celeba_dene33 --deg pr --num_avg_samples 1 --eta 0.78 --etaB 0.17 --timesteps 35 --init_timestep 300
```

To repeat the experiments, pre-trained models should be downloaded from:
[https://drive.google.com/drive/folders/1FK056aWoTSjMJQ6ExjDnj_WlhUI5IUIW?usp=sharing](https://drive.google.com/drive/folders/1FK056aWoTSjMJQ6ExjDnj_WlhUI5IUIW?usp=sharing)
Download the "exp" folder, and, move it to this repo.
The experiments in the project report are also given under "exp/image_samples" folder.


You can refer to:
[https://github.com/bahjat-kawar/ddrm](https://github.com/bahjat-kawar/ddrm) 
