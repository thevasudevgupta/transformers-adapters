# Optimizing Adapters for  Neural Machine Translation

This repositary hosts my experiments for the project, I did with OffNote Labs.  **Blog post summarizing my project can be found** [**here**](https://medium.com/offnote-labs/build-a-model-which-can-translate-multiple-indian-languages-to-english-very-efficiently-reduce-55375fb0e1ea)

## Content

mBART code is taken from **🤗Transformers** & I modified it to add adapters. I made changes to only this [folder](src/transformers/models/bart/) from 🤗Transformers. Rest of 🤗Transformers code is untouched.

Adapters specific code can be found [here](src/transformers/models/bart/adapter_modeling.py). Other boilerplate code can be found [here](src/). All the checkpoints can be found [here](https://drive.google.com/drive/folders/1NjzNfwYO4Hx2yNXyMLHVLgG4GKbHwTB4?usp=sharing).

**Using this project**

```python
# run this from src/ directory
from transformers import MBartForConditionalGeneration, MBartTokenizer

adapter_args = {
    "dec_ffn_adapter": True,
    "enc_self_attn_adapter": True,
    "enc_tok_embed_adapter": True,
    "dec_tok_embed_adapter": True,
    "enc_self_attn_adapter_config": AdapterConfig(input_size=1024),
    "dec_ffn_adapter_config": AdapterConfig(input_size=1024),
    "enc_tok_embed_adapter_config": AdapterConfig(input_size=1024),
    "dec_tok_embed_adapter_config": AdapterConfig(input_size=1024),
}

model = MBartForConditionalGeneration.from_pretrained("vasudevgupta/offnote-mbart-adapters-bhasha")
model.add_adapter_(**adapter_args)
model.load_adapter()

# model is ready for inference just like any other 🤗Transformers
```

**Checkpoints**

| Model details                                                                | checkpoint | BLEU  |
|------------------------------------------------------------------------------|------------|-------|
| Adapters trained on Bhasha Hindi->English with base model `mbart-large-cc25` | https://huggingface.co/vasudevgupta/offnote-mbart-adapters-bhasha/resolve/main/adapters-hin-eng.pt | 25.11 |
| Adapters trained on Bhasha Guj->English with base model `mbart-large-cc25` | https://huggingface.co/vasudevgupta/offnote-mbart-adapters-bhasha/resolve/main/adapters-guj-eng.pt |  |

**Reproducing our results**

```shell
# downloading & unzipping dataset
wget http://preon.iiit.ac.in/~jerin/resources/datasets/pib_v1.3.tar.gz
unzip pib_v1.3.tar.gz

# install requirements
pip3 install -r requirements.txt
cd src
```

```shell
# Reproducing our results

# for normal fine-tuning
python3 train.py --config "main"

# for training best-adapters case on hin-eng
python3 train.py --config "best_adapters_hin"

# for training best-adapters case on hin-eng
python3 train.py --config "best_adapters_guj"

# for inference & bleu score logging over test, run this (after changing config)
python3 infer.py
```