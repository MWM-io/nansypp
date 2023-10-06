# Backbone

## R&D findings

### Data quality

The audio quality of samples in training dataset should be a prior focus before scaling dataset to boost synthesized audio quality. In particular, noise should be avoided as much as possible and audio should be in the highest resolution possible (big difference between 16khz and 44khz trainings).

### Training stability

Addition of gradient clipping was crucial to counter exploding gradients mid-training. Data quality and homogeneity seems to have a significant impact on training stability as well and no good data normalization method was found beneficial yet so this could be an interesting field of further research.

### Pitch supervision

Initial trainings were performed with the relative pitch difference loss described in paper and implemented [here](../src/losses/backbone/generator.py#L273) but showed poor performance (erratic loss and pitch mostly far from target). Significant improvement was achieved when replacing this loss with a mean-square-error of predicted pitch against target pitch extracted using [CREPE](https://github.com/marl/crepe), an of-the-shelf pitch estimator. Focusing on habilitating relative pitch difference could also be an intersting field of further research.

### Backbone as upsampler

To better fit the paper, the dataset can be provided input and outut audios from different sampling rates through `input_data_dirs` and `output_data_dirs`. Audio files with same filenames are then matched as (`input`, `target_output`) enabling for upsampling if output samples are of higher resolution. However, this option was experimented but showed poorer performance.

## Experiments

We provide a backbone checkpoint trained using this repo for 200k training steps / 69 hours on a Quadro RTX 5000.
