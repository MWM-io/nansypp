# NANSY++

Unofficial implementation of paper ["NANSY++: Unified Voice Synthesis with Neural Analysis and Synthesis"](https://arxiv.org/abs/2211.09407) in Pytorch Lightning following guidelines from [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) and drawing inspiration from [revsic unofficial implementation](https://github.com/revsic/torch-nansypp).

We focus on 2 subtasks:

- voice conversion: [documentation](./doc/BACKBONE_DOCUMENTATION.md)
- text-to-speech: [documentation](./doc/TTS_DOCUMENTATION.md)

## ⚙️ Setup

```bash
git clone --recurse-submodules https://github.com/MWM-io/nansypp.git
cd nansypp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
