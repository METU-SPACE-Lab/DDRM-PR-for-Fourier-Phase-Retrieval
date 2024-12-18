import torch
import os
import scipy.io as scipy_io
import re
import matplotlib.pylab as plt
import PIL.Image as Image
import numpy as np
import cupy as cp
import mat73
import cv2
from numpy import linalg as LA
import torch
import numpy as np
import math
import argparse
import torchvision
import os
from argparse import Namespace
import skimage.metrics
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import math
import torchvision
from argparse import Namespace
import numpy as np
import cupy as cp
from numpy import linalg as LA
import mat73
import imageio
import matplotlib.pyplot as plt
import time
import skimage.metrics

import h5py
import cupy as cp
import numpy as np
import torch
from numpy.linalg import norm as LA
import skimage.metrics
import h5py
from tqdm import tqdm
import cv2

# Global Variables
A_matrix = cp.load("exp/empirical_data/A_matrix.npy")
A_adj_matrix = A_matrix.T.conj()

A = lambda x: A_matrix @ x.flatten(order="F")
A_adj = lambda y: A_adj_matrix @ y.flatten(order="F")


# Fienup Phase Retrieval
def fienup_phase_retrieval(
    mag, beta=0.9, steps=200, mode="hybrid", verbose=True, x_init=None
):
    assert beta > 0, "Step size must be a positive number"
    assert steps > 0, "Steps must be a positive number"
    assert mode in [
        "input-output",
        "output-output",
        "hybrid",
    ], "Mode must be 'input-output', 'output-output' or 'hybrid'"

    mag = cp.array(mag)
    if x_init is not None:
        x_init = cp.array(x_init)

    # Initialize y_hat with random phase if no x_init is provided
    if x_init is None:
        y_hat = mag * cp.exp(1j * 2 * cp.pi * cp.random.rand(*mag.shape))
    else:
        y_hat = mag * cp.exp(1j * cp.angle(A(x_init).reshape(mag.shape, order="F")))

    x = cp.zeros_like(mag, dtype=cp.float32)
    x_p = None

    for i in tqdm(range(1, steps + 1)):
        if i % 100 == 0 and verbose:
            print(f"Step {i} of {steps}")

        # Inverse operation
        y = cp.real(A_adj(y_hat))

        if x_p is None:
            x_p = y if x_init is None else x_init
        else:
            x_p = x

        if mode in ["output-output", "hybrid"]:
            x = y

        # Enforce nonnegativity constraint
        indices = y < 0
        if mode in ["hybrid", "input-output"]:
            x[indices] = x_p[indices] - beta * y[indices]
        elif mode == "output-output":
            x[indices] = y[indices] - beta * y[indices]

        # Forward operation
        x_hat = A(x)

        # Reshape x_hat to match mag's shape before enforcing Fourier constraints
        y_hat = mag * cp.exp(1j * cp.angle(x_hat.reshape(mag.shape, order="F")))

    return cp.asnumpy(x)


# Random Initializations
def random_best(magnitudes_oversampled):
    resid_best = float("inf")
    x_init_best = None

    for _ in range(5):
        cp.random.seed(2023)
        result = cp.array(
            fienup_phase_retrieval(magnitudes_oversampled, steps=10, verbose=False)
        )

        # Reshape the result of A(result) to match magnitudes_oversampled
        resid = LA(
            cp.asnumpy(magnitudes_oversampled)
            - cp.asnumpy(
                cp.abs(A(result).reshape(magnitudes_oversampled.shape, order="F"))
            ),
            2,
        )
        if resid < resid_best:
            resid_best = resid
            x_init_best = result

    return x_init_best


# HIO Stage
def hio_stage(image_full_X_test):
    print("EMPIRIK-HIOSTAGE")
    image_full, Y_test = image_full_X_test

    x_init_best = random_best(Y_test)

    result = fienup_phase_retrieval(
        Y_test, steps=300, x_init=x_init_best, verbose=False
    )

    print(
        "result", result.shape, result.min(), result.max(), result.mean(), result.dtype
    )

    # Reshape the result to 64x64
    image_iter = np.reshape((result / 30).clip(0, 255), (64, 64), order="F")

    print(
        "image_full",
        image_full.shape,
        image_full.min(),
        image_full.max(),
        image_full.mean(),
        image_full.dtype,
    )
    print(
        "image_iter",
        image_iter.shape,
        image_iter.min(),
        image_iter.max(),
        image_iter.mean(),
        image_iter.dtype,
    )

    image_iter_flipped = np.flip(image_iter).copy()

    if skimage.metrics.peak_signal_noise_ratio(
        image_full / 255, image_iter_flipped / 255
    ) > skimage.metrics.peak_signal_noise_ratio(image_full / 255, image_iter / 255):
        image_iter = image_iter_flipped

    image_iter = cv2.resize(image_iter, (256, 256))
    print(
        "-image_iter",
        image_iter.shape,
        image_iter.min(),
        image_iter.max(),
        image_iter.mean(),
        image_iter.dtype,
    )
    image_iter = np.repeat(np.expand_dims(image_iter, axis=0), 3, axis=0)

    output = (
        torch.tensor(
            image_iter, device=torch.device("cuda"), dtype=torch.float32
        ).unsqueeze(0)
        / 255
        * 2
        - 1
    )

    print(
        "output", output.shape, output.min(), output.max(), output.mean(), output.dtype
    )

    return output


# PR Encode
def pr_encode(image_full, alpha_=3):
    print("EMPIRIK-PRENCODE")

    with h5py.File("exp/empirical_data/YH_squared_test.mat", "r") as f:
        YH_test = f["YH_squared_test"][:]

    Y_test = np.sqrt(YH_test[:, 1].reshape(4 * 64, 4 * 64, order="F"))

    # Ax =? Y_test
    image_full_tensor = image_full
    image_full = (image_full_tensor[0, 0].cpu().numpy() + 1) / 2 * 255
    image_full_64 = cv2.resize(image_full, (64, 64))
    print(
        "image_full_64", image_full_64.min(), image_full_64.max(), image_full_64.mean()
    )

    # norm of A(image_full_64) - Y_test
    calculated_y = cp.asnumpy(
        cp.abs(A(cp.array(image_full_64 / 255))).reshape((4 * 64, 4 * 64), order="F")
    )
    real_y = Y_test

    print(calculated_y)
    print(Y_test)
    print(calculated_y.min(), calculated_y.max(), calculated_y.mean())
    print(real_y.min(), real_y.max(), real_y.mean())
    # save images real_y, calculated_y
    plt.imsave("real_y.png", real_y, cmap="gray")
    plt.imsave("calculated_y.png", calculated_y, cmap="gray")

    return image_full_64, calculated_y  # Y_test


# always jd(je())
# JE Function
def je(image_full):
    """
    Prepares the data required for phase retrieval using the A function.
    Extracts the first channel of the image_full and reshapes it.
    """
    print("EMPIRIK-JE")

    # Use A function to process image_full's first channel
    image_full_tensor = image_full
    print("image_full_tensor", image_full_tensor.shape)
    image_full = (image_full_tensor[0, 0].cpu().numpy() + 1) / 2 * 255
    # resize 64x64
    image_full_64 = cv2.resize(image_full, (64, 64))

    # Perform Fourier transform using A
    X_test = cp.abs(A(cp.array(image_full_64))).reshape((4 * 64, 4 * 64))

    return image_full_64, X_test


# JD Function
def jd(je_output):
    """
    Performs phase retrieval using Fienup's algorithm without random initialization.
    Processes the output from je and returns a 3-channel batch tensor.
    """
    print("EMPIRIK-JD")
    image_full, X_test = je_output
    print("image_full", image_full.shape, image_full.min(), image_full.max())
    print("X_test", X_test.shape, X_test.min(), X_test.max())

    # Perform Fienup phase retrieval
    result = fienup_phase_retrieval(
        X_test, steps=1000, x_init=image_full, verbose=False
    )

    # Post-process the result
    image_iter = result

    # Reshape the result to 64x64
    image_iter = np.reshape(image_iter, (64, 64))

    print("result", result.shape, result.min(), result.max(), result.dtype)
    print(
        "image_full",
        image_full.shape,
        image_full.min(),
        image_full.max(),
        image_full.dtype,
    )

    # Flip the reconstructed image if necessary
    image_iter_flipped = np.flip(image_iter).copy()

    if skimage.metrics.peak_signal_noise_ratio(
        image_full, image_iter_flipped
    ) > skimage.metrics.peak_signal_noise_ratio(image_full, image_iter):
        image_iter = image_iter_flipped

    # Repeat the result across 3 channels
    image_iter = cv2.resize(image_iter, (256, 256))
    image_iter = np.repeat(np.expand_dims(image_iter, axis=0), 3, axis=0)

    # Normalize and format the output as a batch tensor
    output = (
        torch.tensor(
            image_iter, device=torch.device("cuda"), dtype=torch.float32
        ).unsqueeze(0)
        / image_iter.max()
        * 2
        - 1
    )

    print("output", output.shape, output.min(), output.max(), output.dtype)

    return output
