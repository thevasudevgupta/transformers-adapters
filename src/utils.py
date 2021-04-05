# __author__ = 'Vasudev Gupta'
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def build_seqlen_table(tokenizer, src, tgt, tr_src, tr_tgt, val_src, val_tgt):
     # hin complete data
     lens = [len(tokenizer.tokenize(s)) for s in src]
     src_comp = {'max': np.max(lens), 'avg': np.mean(lens), 'min': np.min(lens)}
    
     # eng complete data
     lens = [len(tokenizer.tokenize(t)) for t in tgt]
     tgt_comp = {'max': np.max(lens), 'avg': np.mean(lens), 'min': np.min(lens)}
    
     # hin train data
     lens = [len(tokenizer.tokenize(s)) for s in tr_src]
     src_tr = {'max': np.max(lens), 'avg': np.mean(lens), 'min': np.min(lens)}
    
     # hin val data
     lens = [len(tokenizer.tokenize(s)) for s in val_src]
     src_val = {'max': np.max(lens), 'avg': np.mean(lens), 'min': np.min(lens)}
    
     # eng train data
     lens = [len(tokenizer.tokenize(t)) for t in tr_tgt]
     tgt_tr = {'max': np.max(lens), 'avg': np.mean(lens), 'min': np.min(lens)}
    
     # eng val data
     lens = [len(tokenizer.tokenize(t)) for t in val_tgt]
     tgt_val = {'max': np.max(lens), 'avg': np.mean(lens), 'min': np.min(lens)}
    
     columns = ['src-complete', 'src-train', 'src-val', 'tgt-complete', 'tgt-train', 'tgt-val']
     data = [[src_comp[k], src_tr[k], src_val[k], tgt_comp[k], tgt_tr[k], tgt_val[k]] for k in ['max', 'avg', 'min']]
    
     return data, columns

@torch.no_grad()
def predictor(bart, tokenizer, lists_src, lists_tgt, pred_max_length, src_lang='hi_IN', device=torch.device("cuda")):
     pred = []
     val_data = []
     tgt = []

     bart.to(device)
     bart.eval()

     bar = tqdm(zip(lists_src, lists_tgt), desc="predicting ... ", leave=True)
     for s, t in bar:
         batch =  tokenizer.prepare_seq2seq_batch(src_texts=s, src_lang=src_lang)

         for k in batch:
            batch[k] = torch.tensor(batch[k])
            batch[k] = batch[k].to(device)

         out = bart.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"], max_length=pred_max_length)
         translation = tokenizer.batch_decode(out, skip_special_tokens=True)
         
         val = list(zip(s, t, translation))
         val_data.extend(val)

         pred.extend(translation)
         tgt.extend(t)

     return val_data, pred, tgt

def read_prepare_data(args):

    # with open("data/itr.txt") as file1:
    #     data = file1.readlines()

    # tgt = [d.split("\t")[0] for d in data]
    # src = [d.split("\t")[1] for d in data]

    with open(args.tgt_file) as file1, open(args.src_file) as file2:
        tgt = file1.readlines()
        src = file2.readlines()
    print('total size of data (src, tgt): ', f'({len(src)}, {len(tgt)})')
    tr_src, val_src, tr_tgt, val_tgt = train_test_split(src, tgt, test_size=args.test_size, random_state=args.random_seed, shuffle=True)
    
    tr_src = tr_src[:args.tr_max_samples]
    tr_tgt = tr_tgt[:args.tr_max_samples]
    val_src = val_src[:args.val_max_samples]
    val_tgt = val_tgt[:args.val_max_samples]

    return tr_src, tr_tgt, val_src, val_tgt, src, tgt
