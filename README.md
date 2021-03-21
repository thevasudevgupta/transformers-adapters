# Transformers with Adapters

## Project Objective

Build a model which can translate multiple Indian languages to english very efficiently.

## Picking initial model

### Why pretrained model?

Indian languages are low resource so we need a model which can understand language structure & related information beforehand nicely so that during training for translation task it can just focus on how to translate and not on understanding each language structure. This way we can work with low data and may get benefitted from other languages (involved during pretraining stage).

### Which pretrained model & Why hard to re-use existing pretrained model?

Pretrained model must support the languages which we want to involve during translation task i.e it must have seen that language during pre-training.

Also pre-trained model should be seq2seq model and hence should be like encoder-decoder architecture.

Using existing pre-trained models is not a easy task. Among so many pretrained models, it's generally hard to decide which one to use. Also many of the pretrained models are english based, so its hard to find model which has seen required language dataset.

Here, lets remain specific to pretrained models for indian-languages, since we are concerned with Indian languages only as the scope for this article.

We had various possible options like IndicBERT (AlBERT like model), mBART and many more. 

**IndicBERT:**

It is a encoder-only model which was trained on multiple Indian languages using BERT objective.
Since our task is of text generation, we need encoder-decoder like model for being able to do this task. One of the way to proceed is to simply stack mBART-Decoder over the IndicBERT model and tune this pair for our task. This experiment is out of scope of this article, but we encourages one to perform these kinda experiments and make better use of pretrained-encoder only models. Some papers claims this kinda approach to be quite effective. You can surely refer [this]([https://arxiv.org/abs/1907.12461](https://arxiv.org/abs/1907.12461)) for reference.

**mBART (mbart-large-cc25)**

Its an encoder-decoder model which is pretrained to denoise multiple languages simultaneously. This way it has captured most of the information of multiple languages and good at tasks involving text generation.

We further introduced adapters in this model at various locations and decided to fine-tune only adapters to adapt the model to translation of particular language pair & preserving almost all of information which it gained during pretraining.

Further described experiments are all about mBART with adapters at appropriate positions.

## Fine-tuning from scratch

Before going deep into adding adapters, lets begin with the most simplest approach i.e. to fine-tune it on each language pair separately. This approach will be very inefficient in terms of memory since we need to save weights of all bi-lingual models.

Eg: lets say, we want translation model for 5 language pairs: hin-eng, guj-eng, tamil-eng, bengali-eng, marathi-eng. mBART finetuned on each dataset will have around 2.5 GB of weights/ language-pair.

So for these 5 language pairs, we will be having 2.5 x 5 = 12.5 GB of weights. Guess how much memory Google translator will take this way (remember it covers around 100 langugages).

We need to find way to increase memory efficiency without compromising the performance on each dataset. We will discuss solutions to this in coming sections.

One more thing to note is that there will be lot of common information among multiple Indian language pairs like language structure of hindi and gujarati are more similar as compared to hindi and eng. This important property can be considered while planning strategy to train the model.

We have reported the result for complete-finetuning in table presented in later section & made a nice comparison with cases of adapter addition.

## Adapters addition

We can simply add small feed-forward like network (commonly called adapters) in-between the layers and let these adapters to get trained instead of training the pretrained weights.

This will limit the language-pair specific weights to be just size of overall feed-forward networks introduced in this step.

**Extending previous example:**

Each adapter weights - 300 MB

5 language pair : 5 x 300 = 1.5 GB

Pretrained weights: 2.5 GB

Total weights: 1.5 + 2.5 = 4 GB

Previously total weights were 12.5 GB and now it is 4 GB. We have reduced 68% of space. Now consider the scenerio when we have 100's of such language pair. Most importantly we are able to do this without compromising our performance. We will compare results in next section.

TOO MUCH EFFICIENT RIGHT!

### Where to add adapters?

Most papers on Adapters suggest to add adapters after feed-forward layer in both encoder and decoder. Some were adding adapters after self-attention layer also.

One of the possible reason of adding adapters after some fixed positions may that where to add adapters is very generic question. One can't really answer this question without testing it; because location of adapters depends on pre-training strategy and how much particular task is related to the pre-training strategy.

Since adapters major role is to adapt each pretrained layer to what it was expected if we would have done complete fine-tuning (approximate way to understand i guess 😉), It should be nice to figure out what all layers are really changing when pretrained model is fine-tuned on our task. We tried to restrict our attention over self-attention / cross attention, ffn, embedding layer. We experimented by freezing some of the layer and tried to visualize how much results are varying after/before freezing that partiular layer. Now based on idea developed from this, we started adding adapters after particular layers. Below table summarizes some of our experiments.

We observed that in case of tuning mBART for translation task, best adapter configuration is adding it after encoder self-attention, decoder feed-forward network & after embedding layer. We observed that adding adapter after embedding layer is very important and can improve the model's performance by significant amount.

### Performance

| Experiment details (20 K samples)                      | Time taken (on Colab T4) | tr_bleu | val_bleu |
|--------------------------------------------------------|--------------------------|---------|----------|
| train dec-ffn-adapter, enc-attn-adapter, embed-adapter | 1h 6m 31s                | 13.991  | 12.54    |
| + enc-ffn-adapter                                      | 1h 8m 40s                | 10.031  | 9.233    |
| + dec-attn-adapter                                     | 1h 10m 34s               | 6.836   | 6.312    |
| + cross-attn-adapter                                   | 1h 9m 52s                | 2.039   | 2.022    |

| Experiment details (50 K samples)                      | Time taken (on Colab T4) | tr_bleu | val_bleu |
|--------------------------------------------------------|--------------------------|---------|----------|
| train dec-ffn-adapter, enc-attn-adapter, embed-adapter | 2h 3m 1s                 | 18.391  | 17.26    |
| + enc-ffn-adapter                                      | 2h 13m 14s               | 3.511   | 3.425    |
| + dec-attn-adapter                                     | 2h 9m 37s                | 8.109   | 7.589    |
| + cross-attn-adapter                                   | 2h 7m 8s                 | 12.233  | 11.42    |

| Experiment details (100 K samples)                     | Time taken (on Colab T4) | tr_bleu | val_bleu |
|--------------------------------------------------------|--------------------------|---------|----------|
| train only embed-adapter                               | 4h 13m 2s                | 15.725  | 15.278   |
| + dec-attn-adapter                                     | 4h 31m 20s               | 21.082  | 19.701   |
| + enc-ffn-adapter                                      | 4h 31m 0s                | 13.932  | 13.126   |
| + dec-ffn-adapter, enc-attn-adapter                    | 4h 27m 25s               | 9.365   | 8.652    |
| Complete tuning                                        | 5h 54m 36s               | 19.833  | 19.049   |

## Deciding hyper-parameters during model training

Unlike transformers which are recommended to train at low learning rate (orders of 1e-5 generally), we observed that adapters takes lot of time to converge at learning rate of this order. After some experiments, we found that for training adapters, learning rate should be in orders of 1e-3 to be able to get convergance in decent time.

Since adapters are saving lots of memory during training as well (gradient calculation are never made (& preserved) for many layers during training), We can have bigger batch sizes and hence fast training on bigger datasets.

### Should we train Adapters & pretrained model simultaneously

We have observed that model performs very worse when adapters and pretrained weights are trained simultaneously. Our hytothesis is that since adapters are introduced after each layer of transformer, weights of every successive layer are getting destroyed because adapters are randomly intialized.

One important question arises here is how to intitalize the adapters. One possible initialization strategy can be to make identity function. We leave this question untouched for the researchers to figure out. We beleive that better initialization of adapters can take pretrained models world to a new era.

## End Notes

- Thanks to **Dr. Nishant Sinha** for guiding me throughout the project and helping me to grow in the world of transformers.
- Thanks to **Hugging Face** team for building such an awesome library for easy & quick experimentation with transformers.

## References

- Simple, Scalable Adaptation for Neural Machine Translation [[Paper](https://www.aclweb.org/anthology/D19-1165.pdf)]
- Pivot-based Transfer Learning for Neural Machine Translation between Non-English Languages [[Paper](https://www.aclweb.org/anthology/D19-1080.pdf)]
- Parameter-efficient Transfer Learning for NLP [[Paper](https://arxiv.org/pdf/1902.00751.pdf)]
- Domain-Adaptation of Pretrained Language Models [[Paper](https://arxiv.org/abs/2004.03354)]
- MAD-X: An Adapter-based Framework for Multi-task Cross-lingual Transfer [[Paper](https://public.ukp.informatik.tu-darmstadt.de/MAD-X/paper.pdf)]
- AdapterHub: A Framework for Adapting Transformers [[Paper](https://arxiv.org/abs/2007.07779)] [[Code](https://github.com/Adapter-Hub/adapter-transformers)]
- Investigate Multiling NMT Representations [[Paper](https://arxiv.org/abs/1909.02197)]
- Massively Multilingual Neural Machine Translation ****[[Paper](https://arxiv.org/pdf/1903.00089.pdf)]
- A study of attention-based Neural Machine Translation models on Indian Languages [[Paper](https://www.aclweb.org/anthology/W16-3717.pdf)]
- bhasha dataset [[Link](http://preon.iiit.ac.in/~jerin/bhasha/)]
- IITB hin-eng parallel dataset corpus [[Link](https://www.cfilt.iitb.ac.in/~parallelcorp/iitb_en_hi_parallel/)]
