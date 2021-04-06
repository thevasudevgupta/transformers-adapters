from transformers import (
    MBartForConditionalGeneration,
    MBartTokenizer
    )

from sacrebleu import corpus_bleu

from dataloader import DataLoader
from utils import (
    predictor, 
    build_seqlen_table, 
    read_prepare_data
    )
from trainer import Trainer
import config
import wandb
import argparse

# python train.py --config "best_adapters_guj"
# python train.py --config "best_adapters_hin"

if __name__ == '__main__':

    # for automating
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="main", help="configurations defined in config.py")
    p_args = parser.parse_args()

    args = getattr(config, p_args.config)
    print(args)

    ## use this for running sweep
    # wandb.init(config=args.__dict__)
    # args = wandb.config
    # print(dict(args))

    tokenizer = MBartTokenizer.from_pretrained(args.tokenizer_id)

    if args.load_dir:
        bart = MBartForConditionalGeneration(args.bart_config)
        print(f"model is loaded from {args.load_dir}")
    else:
        bart = MBartForConditionalGeneration.from_pretrained(args.model_id)
        print(f"model is loaded from {args.model_id}")

    print("====Working on layers freezing====")
    bart.ffn_requires_grad_(args.enc_ffn_grad, args.dec_ffn_grad)
    bart.attn_requires_grad_(args.enc_attn_grad, args.dec_attn_grad, args.cross_attn_grad)
    bart.embed_requires_grad_(args.embed_grad, args.pos_embed_grad)
    bart.norm_requires_grad_(args.enc_norm_grad, args.dec_norm_grad, args.cross_attn_norm_grad)

    print("====Working on adding adapters====")
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

    # initializing adapter with 1
    # with torch.no_grad():
    #     for i in range(len(bart.model.encoder.layers)):
    #         bart.model.encoder.layers[i].adapter_layer.weight = 1
    #     for i in range(len(bart.model.decoder.layers)):
    #         bart.model.decoder.layers[i].adapter_layer.weight = 1

    bart.adapter_requires_grad_(args.enc_ffn_adapter, 
                            args.dec_ffn_adapter,
                            args.cross_attn_adapter,
                            args.enc_self_attn_adapter,
                            args.dec_self_attn_adapter,
                            args.enc_tok_embed_adapter,
                            args.dec_tok_embed_adapter)

    if args.load_adapter_path:
        bart.load_adapter(f"{args.base_dir}/{args.load_adapter_path}")

    if args.load_specific_path:
        bart.load_specific_layers(path=args.load_specific_path, map_location=args.map_location)

    print("====Working on preparing data====")
    tr_src, tr_tgt, val_src, val_tgt, src, tgt = read_prepare_data(args)
    print(len(tr_src), len(tr_tgt), len(val_src), len(val_tgt))
    dl = DataLoader(tr_src, tr_tgt, val_src, val_tgt, tokenizer, args)
    tr_dataset, val_dataset = dl()

    print("====Initiating Trainer====")
    trainer = Trainer(bart, args)
    trainer.fit(tr_dataset, val_dataset)

    if args.save_specific:
        bart.save_specific_layers(path="specific-layers.pt", dec_ffn=True, enc_self_attn=True, tok_embed=True)

    if args.save_adapter_path:
        bart.save_adapter(f"{args.base_dir}/{args.save_adapter_path}", 
                    args.enc_ffn_adapter, 
                    args.dec_ffn_adapter,
                    args.cross_attn_adapter,
                    args.enc_self_attn_adapter,
                    args.dec_self_attn_adapter,
                    args.enc_tok_embed_adapter,
                    args.dec_tok_embed_adapter)

    # trainer.histogram_params(args.tb_params)

    # seqlen logging
    data, columns = build_seqlen_table(tokenizer, src, tgt, tr_src, tr_tgt, val_src, val_tgt)
    wandb.log({'Sequence-Lengths': wandb.Table(data=data, columns=columns)})

    # bleu keeping number of samples in training and validation same
    indices = range(0, len(val_src), args.batch_size)

    src = [tr_src[start:args.batch_size+start] for start in indices]
    tgt = [tr_tgt[start:args.batch_size+start] for start in indices]
    print(f"training results over ({len(src)*args.batch_size}, {len(tgt)*args.batch_size}) ..", end=" ")
    tr_data, pred, tgt = predictor(trainer.model, tokenizer, src, tgt, args.max_pred_length, args.src_lang)
    wandb.log({'tr_predictions': wandb.Table(data=tr_data, columns=['src', 'tgt', 'tgt_pred'])})
    print("||DONE||")

    tr_bleu = corpus_bleu(pred, [tgt]).score
    wandb.log({'tr_bleu': tr_bleu})

    src = [val_src[start:args.batch_size+start] for start in indices]
    tgt = [val_tgt[start:args.batch_size+start] for start in indices]
    print(f"val results over ({len(src)*args.batch_size}, {len(tgt)*args.batch_size}) ..", end=" ")
    val_data, pred, tgt = predictor(trainer.model, tokenizer, src, tgt, args.max_pred_length, args.src_lang)
    wandb.log({'val_predictions': wandb.Table(data=val_data, columns=['src', 'tgt', 'tgt_pred'])})
    print("||DONE||")

    val_bleu = corpus_bleu(pred, [tgt]).score
    wandb.log({'val_bleu': val_bleu})
