# CheMeleon Toxicity Prediction

This document details some (brief) experiments with finetuning the [`CheMeleon`](https://doi.org/10.48550/arXiv.2506.15792) foundation model for the prediction of hepatotoxicity.
Specifically, I'm comparing training Chemprop directly and Chemprop with CheMeleon, both of which using either **direct** training on [DILIrank](https://doi.org/10.1016/j.drudis.2016.02.015) or **pretraining** on [MitoTox](https://doi.org/10.1021/acs.chemrestox.3c00086) and then **finetuning** on DILIrank.

All model training was performed via the Chemprop Command Line Interface, with a small AI-generated Python script for post-processing.
One can easily reproduce these results by running `pip install git+https://github.com/chemprop/chemprop.git@main` (or, if you are reading this after Chemprop 2.2.1 comes out, just `pip install 'chemprop>=2.2.1'`) and then executing the associated bash scripts.
The required data for training and testing is already included alongside this file.

I have't included any specific models from the literature for comparison because I'm focusing more on the use of pretraining and finetuning.
Just to ground ourselves, though, I found [this study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7702310/#S2) which suggests that some of the best models currently available are roughly 70% accurate on binary hepatotoxicity classification (and lower on multiclass 'severity' classification).
The specifics depend a lot on how you split your data, train your model, etc. - but 70% serves as a starting point!
All models in this study are trained using 70% of the data for training and 10% for validation.
The remaining 20% are withheld for testing, and it's the same 20% for all models to fairly gauge the impact of different training strategies.

## Results

# Binary

Reproduce: `. run_binary.sh`

First and foremost, we might want to predict simply if a molecule is a potential hepatotoxin or not - a simple binary classification problem.
We can model this by directly training a Chemprop model on DILIrank (Chemprop + Direct), directly training a CheMeleon model on DILIrank (CheMeleon + Direct), training a Chemprop model first on MitoTox and then on DILIrank (Chemprop + Pretrained) via finetuning, or training a CheMeleon model first on MitoTox and then on DILIrank (CheMeleon + Pretrained) via finetuning.
All four of these approaches, and the resulting accuracy on the randomly withheld 20% of the data, are shown below:

| Model     | Direct | Pretrained |
|-----------|--------|------------|
| ChemProp  | 70.8   | 73.4       |
| CheMeleon | 70.3   | 68.8       |

As we can see, right off the bat both ChemProp and CheMeleon do a good job modeling this target!
An accuracy of ~70% is reasonable in this context, and further hyperparameter optimization might even yield further improved performance.

What's interesting is the result of pretraining on the MitoTox dataset.
For Chemprop this _improves_ the performance on subsequent DILIrank finetuning, adding a nice ~3% increase to the accuracy.
For CheMeleon, though, the performance actually _decreases_ by ~1.5% when finetuned!
This surprising result indicates that the model does a good job fitting to the pretraining and then struggles to transfer to the new data.
The good news, though, is that regular Chemprop does a good job here, so no need to try the foundation model in the first place.

# Multiclass

Reproduce: `. run.sh`

Now the more challenging case - multiclass prediction.
Rather than just predicting if the molecule is hepatotoxic or not, we will try and predict the _degree_ to which it is hepatotoxic.
We follow the same procedure as above, achieving the following results:

| Model     | Direct | Pretrained |
|-----------|--------|------------|
| ChemProp  | 55.7   | 50.0       |
| CheMeleon | 58.3   | 59.3       |

First and foremost, we are doing much worse on this task than on the other task, which is not totally surprising.
The severity of hepatotoxicity is a 'soft' label, i.e. determined on looser criteria than a binary "toxic or not" label.
From a data perspective, there is also less data when training in this manner - relatively few examples of class 2 (high severity) are available, hampering learning.

When training on this task, CheMeleon delivers a nice ~2.5% accuracy increase over a baseline Chemprop model.
Interestingly, the trends seen in [Binary](#binary) are now reversed when doing pretraining and finetuning.
CheMeleon further improves, adding another 1% accuracy to its performance, whereas Chemprop drops substantially, losing almost 6% of its accuracy.
This possibly indicates that default Chemprop fit very well to the pretraining task and struggled to adapt to the new task, whereas CheMeleon thrived when the task changed from pretraining to finetuning (as shown in the original paper).

# Takeaways

Hepatatoxicty prediction _is possible_ with simple data-driven, 2D machine learning models like Chemprop and CheMeleon!
The use of pretraining data can _sometimes_ help with prediction, though it is not as easy as just raking in as much data as possible.
Careful consideration should be given to the choice of task (binary vs. multiclass) the use of pretraining or transfer learning (see [Notes](#notes) below), and hyperparameter optimization (not done here, but easy with `chemprop hpopt`).
The good news is that all of these are readily addressable via the Chemprop CLI without a hassle!

# Notes

All of the above models which were finetuned from pretrained models used the `--from-foundation` argument in the Chemprop CLI.
This command throws out the pretrained Feedforward Neural Network (FNN) used to map the learned molecular fingerprint to the target when training - this is helpful when finetuning on entirely new tasks, but can interfere with model training (as shown in [Binary](#binary)) when the downstream task is the _same_.
One might consider using the [**Transfer Learning**](https://chemprop.readthedocs.io/en/latest/tutorial/cli/train.html#pretraining-and-transfer-learning) capabilities of Chemprop instead, wherein the FNN is _retained_ between pretraining and finetuning (in this case called transfer learning).

## Walkthrough of Commands

The `run.sh` and `run_binary.sh` scripts contain all the needed code to reproduce these results, but no explanations of where they came from.
Below is an annotated walkthrough of the commands, specifically giving the thought process behind why we do what we do.

Let's train a Chemprop model directly on the dilirank dataset, as one might do:

```bash
chemprop train \
    --data-path dilirank.csv \
    --output-dir chemprop_dilirank \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --target-columns dilirank \
    --task-type multiclass \
    --multiclass-num-classes 3 \
    --loss-function ce \
    --split-type random \
    --split-sizes 0.70 0.10 0.20 \
    --data-seed 42
```

Seeking to improve the performance, one might then want to try finetuning a foundation model like CheMeleon on dilirank - this can be done by simply adding the `--from-foundation CheMeleon` argument to your training:

```bash
chemprop train \
    --from-foundation CheMeleon \
    --data-path dilirank.csv \
    --output-dir chemeleon_dilirank \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --target-columns dilirank \
    --task-type multiclass \
    --multiclass-num-classes 3 \
    --loss-function ce \
    --split-type random \
    --split-sizes 0.70 0.10 0.20 \
    --data-seed 42
```

To train a CheMeleon-derived Chemprop model on mitotox and then fine tune on dilirank, run these two commands:

```bash
chemprop train \
    --data-path mitotox.csv \
    --output-dir mitotox_pretrain \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --from-foundation CheMeleon \
    --task-type classification \
    --class-balance \
    --split-type random \
    --split-sizes 0.90 0.09 0.01 \
    --data-seed 42

chemprop train \
    --data-path dilirank.csv \
    --output-dir mitotox_pretrain_dilirank_finetune \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --from-foundation mitotox_pretrain/model_0/best.pt \
    --task-type multiclass \
    --multiclass-num-classes 3 \
    --split-type random \
    --split-sizes 0.70 0.10 0.20 \
    --data-seed 42
```

One could also imagine pretraining regular Chemprop on mitotox and then finetuning on dilirank:

```bash
chemprop train \
    --data-path mitotox.csv \
    --output-dir mitotox_pretrain_chemprop \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --task-type classification \
    --class-balance \
    --split-type random \
    --split-sizes 0.90 0.09 0.01 \
    --data-seed 42

chemprop train \
    --data-path dilirank.csv \
    --output-dir mitotox_pretrain_chemprop_dilirank_finetune \
    --pytorch-seed 42 \
    --smiles-columns smiles \
    --from-foundation mitotox_pretrain_chemprop/model_0/best.pt \
    --task-type multiclass \
    --multiclass-num-classes 3 \
    --split-type random \
    --split-sizes 0.70 0.10 0.20 \
    --data-seed 42
```
