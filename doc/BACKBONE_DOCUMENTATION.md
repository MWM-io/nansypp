# Backbone

## 📚 Quickstart

Prior to any of the following steps, dependencies should be installed as listed in `requirements.txt`.

### Data collect

- Demand: noise dataset used for data augmentation. Can be downloaded [here](https://zenodo.org/record/1227121).
- HifiTTS: high-resolution multi-speaker english dataset used here as baseline. Can be downloaded [here](https://www.openslr.org/109/).

### Data preprocessing

Training dataset and noise dataset audio samples should be decoded and placed in a directory using the following command:

```bash
python src/data/preprocessing/decode.py -i <input_directory> -o <output_directory> -sr <sample_rate>
```

Resulting decoded audio directory paths should be placed in configuration file in place of `noise_dir` and `input_data_dirs`.

### Training

Training can be launched using the following command:

```bash
python src/train/backbone.py --config-name=hifitts +trainer.devices=<list_of_gpu_ids>
```

The configuration name should refer to a Hydra config in the `configs/backbone` folder (YAML file).

### Checkpoint

Run `download_backbone_ckpt.py` that will download a checkpoint we trained using this repository for 200k training-steps and will place it in the right directory so that following inference and app work smoothly.

```bash
python src/utilities/download_checkpoints.py
```

### Inference

An inferencer class is provided in source code and can be called from command-line as follows:

```bash
python src/inference/backbone.py \
<experiment_directory> \
<checkpoint_filename> \
<device> \
<source_audio_path> \
<target_audio_path> \
<output_directory>
```

Example:

```bash
python src/inference/backbone.py \
"static/runs/runs_backbone/hifitts/2023-09-29_16-22-28" \
"opt-steps=step=400000.ckpt" \
"cuda:0" \
"static/samples/vctk/p225_001.wav" \
"static/samples/vctk/p226_002.wav" \
"static/tmp"
```

### Streamlit app

```bash
streamlit run app/reconstruction_and_vc.py --server.port <port_number>
```

### Logs

Along training you can visualize logs using the following command:

```bash
tensorboard --logdir=static/runs/runs_backbone --bind_all --port <port_number>
```

Here is a screenshot of our tensorboard at the end of a 200k training-steps training launched with this repo, following the above guidelines and which results are displayed in a [following section](README.md#🎧-results):
![](./doc/images/os_backbone_tensorboard.png)

## 🔬 R&D

Observations and key R&D results are detailed [here](./BACKBONE_OBSERVATIONS.md).

## 🎧 Results

Results from checkpoints trained with this repo are showcased on [this Notion page](https://swamp-galliform-240.notion.site/Demo-page-for-NANSY-open-source-repo-b38c9ed2722140bf94c3af454e541d37).
