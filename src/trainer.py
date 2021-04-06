# __author__ = 'Vasudev Gupta'

import torch
from torch_trainer import TorchTrainer
import os
from huggingface_hub import ModelHubMixin


class Trainer(TorchTrainer):

    def __init__(self, model, args):

        self.model = model
        self.lr = args.lr
        self.tb_grads = args.tb_grads
        self.args = args

        # self.enc_adapter_config = args.enc_adapter_config
        # self.dec_adapter_config = args.dec_adapter_config

        super().__init__(args)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):

        for k in batch:
          batch[k] = torch.tensor(batch[k])
          batch[k] = batch[k].to(self.device)

        # with torch.cuda.amp.autocast((self.precision=='mixed_16')):
        out = self.model(**batch, return_dict=True)

        loss = out["loss"].mean()

        return loss

    def validation_step(self, batch):

        for k in batch:
          batch[k] = torch.tensor(batch[k])
          batch[k] = batch[k].to(self.device)

        with torch.no_grad():
            # with torch.cuda.amp.autocast((self.precision=='mixed_16')):
            out = self.model(**batch, return_dict=True)
            loss = out["loss"].mean()

        return loss

    def training_epoch_end(self, epoch, losses):
        # if epoch == 4:
        #     self.model.add_adapter_(False, True, 
        #             self.enc_adapter_config, 
        #             self.dec_adapter_config)
        #     self.model.activate_adapter_(False, True)
        #     self.model.to(self.device)
        #     print("dec-adapter trainable status = True")
        # self.save_pretrained(f"{self.base_dir}/HF/mbart-hin-eng-{epoch}")

        if self.args.save_adapter_path:
            self.model.save_adapter(f"{self.args.base_dir}/{self.args.save_adapter_path}-{epoch}.pt", 
                        self.args.enc_ffn_adapter,
                        self.args.dec_ffn_adapter,
                        self.args.cross_attn_adapter,
                        self.args.enc_self_attn_adapter,
                        self.args.dec_self_attn_adapter,
                        self.args.enc_tok_embed_adapter,
                        self.args.dec_tok_embed_adapter)
            self.save_training_state_dict(self.base_dir)

        if self.args.finetuned_id is not None:
            save_dir = os.path.join(self.base_dir, f"transformers-adapters-e{epoch}")
            self.save_pretrained(save_dir)
            try:
                ModelHubMixin.push_to_hub(save_dir, model_id=self.args.finetuned_id, commit_message=f"add epoch-{epoch}")
            except Exception as e:
                print(e)

    def save_pretrained(self, path: str):
        print('saving weights in HF format')
        module = self.model.module if hasattr(self.model, "module") else self.model
        module.save_pretrained(path)
