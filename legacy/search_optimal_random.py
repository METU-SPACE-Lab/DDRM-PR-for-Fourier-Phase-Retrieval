import subprocess
import numpy as np
import re
import random

random_parameters = []
for eta in np.arange(0.7, 1.0, 0.08):
    for etaB in np.arange(0.07, eta * 2 / 3, 0.1):
        for num_avg_samples in range(1, 21, 3):
            for timesteps in range(15, 36, 3):
                if random.random() < 0.1:
                    random_parameters.append([eta, etaB, num_avg_samples, timesteps])
print(len(random_parameters))

print("eta, \t etaB, \t num_avg_samples, \t timesteps: \t  \t hio \t \t diff")

with open('allcomb.txt', 'w') as f:
    f.write("\n".join(["asdasd", "asdasd"]))

# for eta, etaB, num_avg_samples, timesteps in random_parameters:
#     command = f"python main.py --ni --config celeba_pr.yml --doc celeba -i celeba --deg pr --num_avg_samples {num_avg_samples} --eta {eta} --etaB {etaB} --timesteps {timesteps}"

#     hio_psnr = 0
#     diff_psnr = 0
#     MONTE_CARLO = 1
#     for monte_carlo_step in range(MONTE_CARLO):
#         output = subprocess.getoutput(command)

#         hio_psnr += float(
#             re.findall("Total Average PSNR of JPEG: (.*)", output)[0]
#         )
#         diff_psnr += float(
#             re.findall("Total Average PSNR: (.*)", output)[0]
#         )
#     hio_psnr /= MONTE_CARLO
#     diff_psnr /= MONTE_CARLO
    
#     # print(output)
#     print(
#         f"{eta}, \t {etaB}, \t {num_avg_samples}, \t {timesteps}: \t  \t {hio_psnr} \t \t {diff_psnr}"
#     )