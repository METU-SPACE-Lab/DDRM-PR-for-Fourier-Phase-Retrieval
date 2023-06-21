To repeat the experiments, create conda environment with environment.yml. And, then run  main.py like this:

```
python main.py --ni --config celeba_pr.yml --doc celeba_dene33 -i celeba_dene33 --deg pr --num_avg_samples 1 --eta 0.78 --etaB 0.17 --timesteps 35 --init_timestep 300
```

The images in the project is given under the exp/image_samples folder.

You can refer to:
[https://github.com/bahjat-kawar/ddrm](https://github.com/bahjat-kawar/ddrm) 