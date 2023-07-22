# Noisy Prediction Calibration (2022)

This repository contains implementation of the source code for the paper (Bae,
HeeSun, et al. "From noisy prediction to true label: Noisy prediction
calibration via generative model." International Conference on Machine
Learning. PMLR, 2022.) in Tensorflow 2. The original PyTorch implementation can
be found at https://github.com/BaeHeeSun/NPC/tree/main.

## Experiments

| Model            | MNIST | MNIST | Fashion-MNIST | Fashion-MNIST | Fashion-MNIST | Fashion-MNIST | Fashion-MNIST | Fashion-MNIST | Fashion-MNIST | Fashion-MNIST | Fashion-MNIST | CIFAR-10 | CIFAR-10 | CIFAR-10 | CIFAR-10 | CIFAR-10 | CIFAR-10 | CIFAR-10 | CIFAR-10 | CIFAR-10 |
|------------------|-------|-------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
|                  | Clean | IDN   | Clean         | SN            | SN            | ASN           | ASN           | IDN           | IDN           | SRIDN         | SRIDN         | Clean    | SN       | SN       | ASN      | ASN      | IDN      | IDN      | SRIDN    | SRIDN    |
|                  | -     | 40%   | -             | 20%           | 80%           | 20%           | 40%           | 20%           | 40%           | 20%           | 40%           | 20%      | 40%      | 20%      | 40%      | -        | 20%      | 80%      | 20%      | 40%      | 20% | 40% | 20% | 40% |
| CE (authors)     | 97.8  | 66.3  | 87.1          | 74.0          | 27.0          | 81.0          | 77.3          | 68.4          | 52.1          | 81.0          | 67.3          | 86.9     | 73.1     | 15.1     | 80.2     | 71.4     | 72.9     | 53.9     | 72.6     | 61.8     |
| w/ NPC (authors) | 98.2  | 89.0  | 88.4          | 84.0          | 35.8          | 85.9          | 86.2          | 82.5          | 74.5          | 81.8          | 69.4          | 89.0     | 80.8     | 17.0     | 84.7     | 78.8     | 80.9     | 59.9     | 74.3     | 64.3     |
| CE (mine)        |       |       |               |               |               |               |               |               |               |               |               |          |          |          |          |          |          |          |          |          |
| w/ NPC (mine)    |       |       |               |               |               |               |               |               |               |               |               |          |          |          |          |          |          |          |          |          |
