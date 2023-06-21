export CUDA_VISIBLE_DEVICES=0
python main.py --ni --config celeba_pr.yml --doc celeba_dene26 -i celeba_dene26 --deg pr --num_avg_samples 1 --eta 1.0 --etaB 0.6 --timesteps 20 --init_timestep 300