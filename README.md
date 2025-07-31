# `chemeleon_tox`

This repo contains some experiments using the CheMeleon foundation model on toxicity prediction.

It is particulary focused on the use of multi-stage pretraining, and transfer learning.

The original CheMeleon model was pretrained on a molecular descriptor prediction task.
When we want to apply it to a new task (like DILI), is it best to (1) directly train it on data (2) pretrain it first, then train it or (3) use transfer learning?

`mitotox_dilirank` compares the usage of pretraining vs. direct training for both Chemprop and CheMeleon.
`seal_et_al_comparison` compares the usage of finetuning and direct training, demonstrating that CheMeleon can achieve state-of-the-art results using this approach.
