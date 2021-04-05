# Optimizing Adapters for  Neural Machine Translation

This repositary hosts my experiments for the project, I did with OffNote Labs.  **Blog post summarizing my project can be found** [**here**](https://medium.com/offnote-labs/build-a-model-which-can-translate-multiple-indian-languages-to-english-very-efficiently-reduce-55375fb0e1ea)

## Content

mBART code is taken from **🤗Transformers** & I modified it to add adapters. I made changes to only this [folder](src/transformers/models/bart/) from 🤗Transformers. Rest of 🤗Transformers code is untouched.

Adapters specific code can be found [here](src/transformers/models/bart/adapter_modeling.py). Other boilerplate code can be found [here](src/). All the checkpoints can be found [here](https://drive.google.com/drive/folders/1NjzNfwYO4Hx2yNXyMLHVLgG4GKbHwTB4?usp=sharing).

**Running this project**

```shell
# install requirements
pip3 install -r requirements.txt
cd src

# change config as per your need & run following for training
python3 train.py --config "main"

# for inference & bleu score logging over test, run this
python3 infer.py
```

**PS:** I tried making the code a bit more readable and I haven't tested it completely so it might break in some places (old one was super messy). If its something major, feel free to put up an issue.
