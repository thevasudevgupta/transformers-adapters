# Optimizing Adapters for  Neural Machine Translation

This repositary hosts my experiments for the project, I did with [OffNote Labs](https://github.com/ofnote).  **Blog post summarizing my project can be found** [**here**](https://medium.com/offnote-labs/build-a-model-which-can-translate-multiple-indian-languages-to-english-very-efficiently-reduce-55375fb0e1ea).

## Details

`mBART` code is taken from **🤗Transformers** & I modified it to add adapters. I made changes to only this [folder](src/transformers/models/bart/) from 🤗Transformers. Rest of 🤗Transformers code is untouched.

Adapters specific code can be found [here](src/transformers/models/bart/adapter_modeling.py). Other boilerplate code can be found [here](src/).

**Checkpoints**

| Model details               | Checkpoint                                                                         | Size   |
|-----------------------------|------------------------------------------------------------------------------------|--------|
| `hin->eng` adapters         | [`adapters-hin-eng`](https://huggingface.co/vasudevgupta/offnote-mbart-adapters-bhasha/resolve/main/adapters-hin-eng.pt) | 104.6 MB |
| `hin->eng` Fine-tuned model | [`mbart-bhasha-hin-eng`](https://huggingface.co/vasudevgupta/mbart-bhasha-hin-eng) | 2.3 GB |
| `guj->eng` adapters         | [`adapters-guj-eng`](https://huggingface.co/vasudevgupta/offnote-mbart-adapters-bhasha/resolve/main/adapters-guj-eng.pt) | 104.6 MB |
| `guj->eng` Fine-tuned model | [`mbart-bhasha-guj-eng`](https://huggingface.co/vasudevgupta/mbart-bhasha-guj-eng) | 2.3 GB |

*Note: Bhasha **guj-eng** dataset has only 59K samples; while **hin-eng** dataset has 260K samples*

Other checkpoints can be found [here](https://drive.google.com/drive/folders/1NjzNfwYO4Hx2yNXyMLHVLgG4GKbHwTB4?usp=sharing).

**Setting Up**

```shell
# install requirements
pip3 install -r requirements.txt
cd src

# downloading & unzipping dataset
wget http://preon.iiit.ac.in/~jerin/resources/datasets/pib_v1.3.tar.gz
unzip pib_v1.3.tar.gz

# Adapters checkpoints (it's just 200 MB) can be downloaded using:
git clone https://huggingface.co/vasudevgupta/offnote-mbart-adapters-bhasha
```

**Using this project**

```python
# run this from src/ directory
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers.models.bart.adapter_modeling import AdapterConfig

# initialize mBART from pre-trained weights
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")

# Now let's add adapters which will enable multilingual translation

# deciding what all adapters to add
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

# add randomly-initialized adapters
model.add_adapter_(**adapter_args)

# loading hin-eng adapters
model.load_adapter("offnote-mbart-adapters-bhasha/adapters-hin-eng.pt", map_location="cpu")

# loading guj-eng adapters
model.load_adapter("offnote-mbart-adapters-bhasha/adapter-guj-eng.pt", map_location="cpu")

# model is ready for inference just like any other 🤗Transformers
```

**Reproducing our results**

```shell
# for normal fine-tuning
python3 train.py --config "main"

# for training with best-adapters case on hin-eng
python3 train.py --config "best_adapters_hin"
# for normal-finetuning on hin-eng
python3 train.py --config "full_train_hin"

# for training with best-adapters case on guj-eng
python3 train.py --config "best_adapters_guj"
# for normal-finetuning on guj-eng
python3 train.py --config "full_train_guj"

# for inference & bleu score logging over test, run this (after changing config)
python3 infer.py
```
