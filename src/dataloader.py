# __author__ = 'Vasudev Gupta'

import torch


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, src: list, tgt: list):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {
            'input_ids': self.src[idx],
            'labels': self.tgt[idx]
        }

class DataLoader(object):

    def __init__(self, tr_src, tr_tgt, val_src, val_tgt, tokenizer, args):

        self.tokenizer = tokenizer

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.max_length = args.max_length
        self.max_target_length = args.max_target_length

        self.tr_src = tr_src
        self.tr_tgt = tr_tgt

        self.val_src = val_src
        self.val_tgt = val_tgt

        self.src_lang = args.src_lang
        self.tgt_lang = args.tgt_lang

    def __call__(self):
        self.setup()
        tr_dataset = self.train_dataloader()
        val_dataset = self.val_dataloader()
        return tr_dataset, val_dataset

    def setup(self):
        self.tr_dataset = CustomDataset(self.tr_src, self.tr_tgt)
        self.val_dataset = CustomDataset(self.val_src, self.val_tgt)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.tr_dataset,
                                            pin_memory=True,
                                            shuffle=True,
                                            batch_size=self.batch_size,
                                            collate_fn=self.collate_fn,
                                            num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                            pin_memory=True,
                                            shuffle=False,
                                            batch_size=self.batch_size,
                                            collate_fn=self.collate_fn,
                                            num_workers=self.num_workers)

    def collate_fn(self, features):
        
        inputs = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        
        batch =  self.tokenizer.prepare_seq2seq_batch(
            src_texts=inputs, src_lang=self.src_lang, tgt_lang=self.tgt_lang, tgt_texts=labels,
            max_length=self.max_length, max_target_length=self.max_target_length)
        
        return batch
