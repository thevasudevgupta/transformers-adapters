# __author__ = 'Vasudev Gupta'

from transformers import (
    MBartForConditionalGeneration,
    MBartTokenizer
    )
from transformers.models.bart.adapter_modeling import AdapterConfig 
import wandb

from sacrebleu import corpus_bleu

from utils import (
    predictor,
    read_prepare_data
    )
from dataclasses import dataclass, field


@dataclass
class Config:

    tokenizer_id: str = 'facebook/mbart-large-cc25'
    model_id: str = 'facebook/mbart-large-cc25' # "vasudevgupta/mbart-iitb-hin-eng"
    batch_size: int = 32

    tgt_file: str = "data/parallel/IITB.en-hi.en"
    src_file: str = "data/parallel/IITB.en-hi.hi"

    tr_max_samples: int = 20
    val_max_samples: int = 20
    base_dir: str = "tr_dec-ffn-adapter_enc-attn-adapter_embed-adapter_hin0.5M"

    # data_file: str = "itr.txt"
    test_size: float = .25
    random_seed:  int = 7232114
    max_pred_length: int = 32
    src_lang: str = "hi_IN"

    # control adapter from here
    # manually switch off layers in case you want to freeze
    load_adapter_path: str = None
    enc_ffn_adapter: bool = False
    dec_ffn_adapter: bool = True
    enc_self_attn_adapter: bool = True
    dec_self_attn_adapter: bool = False
    cross_attn_adapter: bool = False
    enc_tok_embed_adapter: bool = True
    dec_tok_embed_adapter: bool = True

    # adapter inside config
    enc_ffn_adapter_config: AdapterConfig = field(repr=False, default=AdapterConfig(input_size=1024))

    dec_ffn_adapter_config: AdapterConfig = field(repr=False, default=AdapterConfig(input_size=1024))

    enc_self_attn_adapter_config: AdapterConfig = field(repr=False, default=AdapterConfig(input_size=1024))

    dec_self_attn_adapter_config: AdapterConfig = field(repr=False, default=AdapterConfig(input_size=1024))

    cross_attn_adapter_config: AdapterConfig = field(repr=False, default=AdapterConfig(input_size=1024))

    dec_tok_embed_adapter_config: AdapterConfig = field(repr=False, default=AdapterConfig(input_size=1024,
                                                                                        add_layer_norm_after=False))

    enc_tok_embed_adapter_config: AdapterConfig = field(repr=False, default=AdapterConfig(input_size=1024,
                                                                                        add_layer_norm_after=False))

if __name__ == '__main__':

    args = Config()
    print(args)

    wandb.init(project="mbart", config=args.__dict__)

    tokenizer = MBartTokenizer.from_pretrained(args.tokenizer_id)

    print(f"model is loaded from {args.model_id}")
    bart = MBartForConditionalGeneration.from_pretrained(args.model_id)

    tr_src, tr_tgt, val_src, val_tgt, src, tgt = read_prepare_data(args)
    print(len(tr_src), len(tr_tgt), len(val_src), len(val_tgt))

    bart.add_adapter_(args.enc_ffn_adapter, 
                args.dec_ffn_adapter,
                args.enc_self_attn_adapter,
                args.dec_self_attn_adapter,
                args.cross_attn_adapter,
                args.enc_tok_embed_adapter,
                args.dec_tok_embed_adapter,
                args.enc_ffn_adapter_config,
                args.dec_ffn_adapter_config,
                args.enc_self_attn_adapter_config,
                args.dec_self_attn_adapter_config,
                args.cross_attn_adapter_config,
                args.enc_tok_embed_adapter_config,
                args.dec_tok_embed_adapter_config)

    if args.load_adapter_path:
        bart.load_adapter(f"{args.base_dir}/{args.load_adapter_path}")

    # bleu keeping number of samples in training and validation same
    indices = range(0, len(val_src), args.batch_size)

    src = [tr_src[start:args.batch_size+start] for start in indices]
    tgt = [tr_tgt[start:args.batch_size+start] for start in indices]
    print(len(src)*args.batch_size, len(tgt)*args.batch_size)
    tr_data, pred, tgt = predictor(bart, tokenizer, src, tgt, args.max_pred_length, args.src_lang)
    wandb.log({'tr_predictions': wandb.Table(data=tr_data, columns=['src', 'tgt', 'tgt_pred'])})

    tr_bleu = corpus_bleu(pred, [tgt]).score
    wandb.log({'tr_bleu': tr_bleu})

    src = [val_src[start:args.batch_size+start] for start in indices]
    tgt = [val_tgt[start:args.batch_size+start] for start in indices]
    print(len(src)*args.batch_size, len(tgt)*args.batch_size)
    val_data, pred, tgt = predictor(bart, tokenizer, src, tgt, args.max_pred_length, args.src_lang)
    wandb.log({'val_predictions': wandb.Table(data=val_data, columns=['src', 'tgt', 'tgt_pred'])})

    val_bleu = corpus_bleu(pred, [tgt]).score
    wandb.log({'val_bleu': val_bleu})
