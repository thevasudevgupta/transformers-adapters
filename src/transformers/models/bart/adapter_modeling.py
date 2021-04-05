import torch
from torch import nn

from ...activations import ACT2FN
from dataclasses import dataclass

"""
USAGE:
    This file is having everything related to Adapters
    Small changes has been made in `modeling_bart.py` for mixing
"""

@dataclass
class AdapterConfig:
    input_size: int # same as d_model
    down_sample: int = None # size of downsampling
    non_linearity: str = "relu"       
    add_layer_norm_before: bool = True
    residual_before_ln: bool = True
    init_bert_weights: bool = True
    add_layer_norm_after: bool = True

class Activation_Function_Class(nn.Module):

    def __init__(self, hidden_act):
        super().__init__()

        self.f = ACT2FN[hidden_act.lower()]

    def forward(self, x):
        return self.f(x)


class Adapter(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.input_size = args.input_size
        self.add_layer_norm_before = args.add_layer_norm_before
        self.add_layer_norm_after = args.add_layer_norm_after
        self.residual_before_ln = args.residual_before_ln

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = args.down_sample
        if self.down_sample is None:
            self.down_sample = self.input_size // 2

        # Linear down projection of the input
        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        self.non_linearity = Activation_Function_Class(args.non_linearity)

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if args.init_bert_weights:
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)

    def forward(self, x, residual_input):
        down = self.adapter_down(x)

        up = self.adapter_up(down)

        output = up

        # todo add brief documentation what that means
        if self.residual_before_ln:
            output = output + residual_input

        # todo add brief documentation what that means
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # todo add brief documentation what that means
        if not self.residual_before_ln:
            output = output + residual_input

        return output, down, up

    @staticmethod
    def init_bert_weights(module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class AdapterLayerMix(object):
    # Inherit encoder-layer / decoder-layer from this class

    def __init__(self):
        # call this method at end of `__init__`
        self.add_adapter_ffn = False
        self.add_adapter_self_attn = False
        self.add_adapter_cross_attn = False

    def adapter_ffn_forward(self, adapter_in, adapter_residual_conn):
        # call this method at end of layer `forward`

        adapter_out = self.adapter_ffn(adapter_in, adapter_residual_conn)
        adapter_out = adapter_out[0]

        return adapter_out

    def adapter_self_attn_forward(self, adapter_in, adapter_residual_conn):
        # call this method at end of layer `forward`

        adapter_out = self.adapter_self_attn(adapter_in, adapter_residual_conn)
        adapter_out = adapter_out[0]

        return adapter_out

    def adapter_cross_attn_forward(self, adapter_in, adapter_residual_conn):

        adapter_out = self.adapter_cross_attn(adapter_in, adapter_residual_conn)
        adapter_out = adapter_out[0]

        return adapter_out

    def add_adapter_ffn_(self, config_ffn):

        self.add_adapter_ffn = True
        self.adapter_ffn = Adapter(config_ffn)

        return "ffn adapter added"

    def add_adapter_self_attn_(self, config_self_attn):

        self.add_adapter_self_attn = True
        self.adapter_self_attn = Adapter(config_self_attn)

        return "self-attn adapter added"

    def add_adapter_cross_attn_(self, config_cross_attn):
        # remember to call it only with decoder

        self.add_adapter_cross_attn = True
        self.adapter_cross_attn = Adapter(config_cross_attn)

        return "cross-attn adapter added"

    def adapter_requires_grad_(self, ffn_grad, self_attn_grad, cross_attn_grad=None):

        m1 = "ffn NOT activated"
        m2 = "self-attn NOT activated"
        m3 = "cross-attn NOT activated"

        if self.add_adapter_ffn:
            m1 = "ffn adapter not activated"
            for param in self.adapter_ffn.parameters():
                param.requires_grad_(ffn_grad)
            if ffn_grad:
                m1 = "ffn adapter activated"
        else:
            m1 = "ffn adapter ADD first"

        if self.add_adapter_self_attn:
            m2 = "self-attn adapter not activated"
            for param in self.adapter_self_attn.parameters():
                param.requires_grad_(self_attn_grad)
            if self_attn_grad:
                m2 = "self-attn adapter activated"
        else:
            m2 = "self-attn adapter ADD first"

        if self.add_adapter_cross_attn:
            m3 = "cross-attn adapter not activated"
            for param in self.adapter_cross_attn.parameters():
                param.requires_grad_(cross_attn_grad)
            if cross_attn_grad:
                m3 = "cross-attn adapter activated"
        else:
            m3 = "cross-attn adapter ADD first"

        return m1, m2, m3

class MixAdapterEncDec(object):

    def __init__(self):
        self.add_adapter_tok_embed = False

    def add_adapter_tok_embed_(self, config_tok_embed):

        self.add_adapter_tok_embed = True
        self.adapter_tok_embed = Adapter(config_tok_embed)

        return "tok-embed adapter added"

    def adapter_tok_embed_forward(self, adapter_in, adapter_residual_conn):

        adapter_out = self.adapter_tok_embed(adapter_in, adapter_residual_conn)
        adapter_out = adapter_out[0]

        return adapter_out

    def adapter_requires_grad_(self, tok_embed):

        m = "tok-embed adapter ADD first"

        if self.add_adapter_tok_embed:
            m = "tok-embed adapter not activated"
            for param in self.adapter_tok_embed.parameters():
                param.requires_grad_(tok_embed)
            if tok_embed:
                m = "tok-embed adapter activated"
        else:
            m = "tok-embed adapter ADD first"

        return m

class MixAdapterBFCG(object):

    def __init__(self):
        """Inherit BFCG from this this class"""

    def add_adapter_(self,
                    enc_ffn_adapter: bool, 
                    dec_ffn_adapter: bool,
                    enc_self_attn_adapter: bool,
                    dec_self_attn_adapter: bool,
                    cross_attn_adapter: bool,
                    enc_tok_embed_adapter: bool,
                    dec_tok_embed_adapter: bool,
                    enc_ffn_adapter_config: AdapterConfig,
                    dec_ffn_adapter_config: AdapterConfig,
                    enc_self_attn_adapter_config: AdapterConfig,
                    dec_self_attn_adapter_config: AdapterConfig,
                    cross_attn_adapter_config: AdapterConfig,
                    enc_tok_embed_adapter_config: AdapterConfig,
                    dec_tok_embed_adapter_config: AdapterConfig):

        m1 = "encoder ffn adapter NOT added"
        m2 = "encoder self-attn adapter NOT added"
        m3 = "decoder ffn adapter NOT added"
        m4 = "decoder self-attn adapter NOT added"
        m5 = "cross-attn adapter NOT added"
        m6 = "encoder tok-embed adapter NOT added"
        m7 = "decoder tok-embed adapter NOT added"

        if enc_ffn_adapter:
            num = len(self.model.encoder.layers)
            for i in range(num):
                m1 = self.model.encoder.layers[i].add_adapter_ffn_(enc_ffn_adapter_config)
                m1 = "encoder " + m1

        if enc_self_attn_adapter:
            num = len(self.model.encoder.layers)
            for i in range(num):
                m2 = self.model.encoder.layers[i].add_adapter_self_attn_(enc_self_attn_adapter_config)
                m2 = "encoder " + m2

        if dec_ffn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                m3 = self.model.decoder.layers[i].add_adapter_ffn_(dec_ffn_adapter_config)
                m3 = "decoder " + m3

        if dec_self_attn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                m4 = self.model.decoder.layers[i].add_adapter_self_attn_(dec_self_attn_adapter_config)
                m4 = "decoder " + m4

        if cross_attn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                m5 = self.model.decoder.layers[i].add_adapter_cross_attn_(cross_attn_adapter_config)

        if enc_tok_embed_adapter:
            m6 = self.model.encoder.add_adapter_tok_embed_(enc_tok_embed_adapter_config)
            m6 = "encoder " + m6

        if dec_tok_embed_adapter:
            m7 = self.model.decoder.add_adapter_tok_embed_(dec_tok_embed_adapter_config)
            m7 = "decoder " + m7

        print("==========Adapter status============================")
        print(m1, "\n", m2, "\n", m3, "\n", m4, "\n", m5, "\n", m6, "\n", m7)
        print("====================================================")

    def adapter_requires_grad_(self,
                    enc_ffn_adapter: bool,
                    dec_ffn_adapter: bool,
                    cross_attn_adapter: bool,
                    enc_self_attn_adapter: bool,
                    dec_self_attn_adapter: bool,
                    enc_tok_embed_adapter: bool,
                    dec_tok_embed_adapter: bool):

        num = len(self.model.encoder.layers)
        for i in range(num):
            m1, m2, _ = self.model.encoder.layers[i].adapter_requires_grad_(enc_ffn_adapter, enc_self_attn_adapter)
            m1, m2 = "encoder " + m1, "encoder " + m2

        num = len(self.model.decoder.layers)
        for i in range(num):
            m3, m4, m5 = self.model.decoder.layers[i].adapter_requires_grad_(dec_ffn_adapter, dec_self_attn_adapter, cross_attn_adapter)
            m3, m4, m5 = "decoder " + m3, "decoder " + m4, m5

        m6 = self.model.encoder.adapter_requires_grad_(enc_tok_embed_adapter)
        m7 = self.model.decoder.adapter_requires_grad_(dec_tok_embed_adapter)

        print("==========Adapter activation status==========")
        print(m1, "\n", m2, "\n", m3, "\n", m4, "\n", m5, "\n", m6, "\n", m7)
        print("=============================================")

    def save_adapter(self,
                    path: str,
                    enc_ffn_adapter: bool,
                    dec_ffn_adapter: bool,
                    cross_attn_adapter: bool,
                    enc_self_attn_adapter: bool,
                    dec_self_attn_adapter: bool,
                    enc_tok_embed_adapter: bool,
                    dec_tok_embed_adapter: bool):

        state_dict = self.state_dict()
        saving_keys = []

        if enc_ffn_adapter:
            num = len(self.model.encoder.layers)
            for i in range(num):
                k = f"model.encoder.layers.{i}.adapter_ffn"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if dec_ffn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                k = f"model.decoder.layers.{i}.adapter_ffn"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])
        
        if cross_attn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                k = f"model.decoder.layers.{i}.adapter_cross_attn"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if enc_self_attn_adapter:
            num = len(self.model.encoder.layers)
            for i in range(num):
                k = f"model.encoder.layers.{i}.adapter_self_attn"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if dec_self_attn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                k = f"model.decoder.layers.{i}.adapter_self_attn"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if enc_tok_embed_adapter:
            k = f"model.encoder.adapter_tok_embed"
            saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if dec_tok_embed_adapter:
            k = f"model.decoder.adapter_tok_embed"
            saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        saving = {}
        for k in saving_keys:
            saving.update({k: state_dict[k]})

        if path:
            print(f"saving: {saving.keys()}")
            torch.save(saving, path)

    def load_adapter(self, path: str = None, map_location="cuda:0"):
        # simple loading; saving is very important
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=False)
        print(f'Whatever weights were in {path} are loaded')

class MixTrainableController(object):

    def __init__(self):
        """
        trainable_status = True :: layer will be trained
        trainable_status = False :: layer will be freezed

        trainable_status = None :: This class won't be used
        """

    def ffn_requires_grad_(self, enc_ffn: bool, dec_ffn: bool):

        num = len(self.model.encoder.layers)
        for i in range(num):

            for p in self.model.encoder.layers[i].fc1.parameters():
                p.requires_grad_(enc_ffn)

            for p in self.model.encoder.layers[i].fc2.parameters():
                p.requires_grad_(enc_ffn)

        if enc_ffn:
            m1 = "trainable"
        else:
            m1 = "freezed"

        num = len(self.model.decoder.layers)
        for i in range(num):

            for p in self.model.decoder.layers[i].fc1.parameters():
                p.requires_grad_(dec_ffn)

            for p in self.model.decoder.layers[i].fc2.parameters():
                p.requires_grad_(dec_ffn)

        if dec_ffn:
            m2 = "trainable"
        else:
            m2 = "freezed"

        print(
          """
             |-----------------------------------|
             |    Layer Name    |     Status     |
             |-----------------------------------|
                 Encoder FFN    |    {}       
             |-----------------------------------|
                 Decoder FFN    |    {}       
             |-----------------------------------|
          """.format(m1, m2)
        )

    def attn_requires_grad_(self, enc_attn: bool, dec_attn: bool, cross_attn: bool):

        # encoder attention
        num = len(self.model.encoder.layers)
        for i in range(num):
                
            for p in self.model.encoder.layers[i].self_attn.parameters():
                p.requires_grad_(enc_attn)

        if enc_attn:
            m1 = "trainable"
        else:
            m1 = "freezed"

        # decoder attention
        num = len(self.model.decoder.layers)
        for i in range(num):

            for p in self.model.decoder.layers[i].self_attn.parameters():
                p.requires_grad_(dec_attn)

        if dec_attn:
            m2 = "trainable"
        else:
            m2 = "freezed"

        # cross attention
        num = len(self.model.decoder.layers)
        for i in range(num):

            for p in self.model.decoder.layers[i].encoder_attn.parameters():
                p.requires_grad_(cross_attn)

        if cross_attn:
            m3 = "trainable"
        else:
            m3 = "freezed"

        print(
          """
             |-----------------------------------|
             |    Layer Name     |     Status    |
             |-----------------------------------|
                 Encoder ATTN    |    {}       
             |-----------------------------------|
                 Decoder ATTN    |    {}       
             |-----------------------------------|
                 Enc-Dec ATTN    |    {}       
             |-----------------------------------|
          """.format(m1, m2, m3)
        )

    def embed_requires_grad_(self, embed_grad: bool, pos_embed_grad: bool):

        for p in self.model.shared.parameters():
            p.requires_grad_(embed_grad)

        if embed_grad:
            m1 = "trainable"
        else:
            m1 = "freezed"

        for p in self.model.encoder.embed_positions.parameters():
            p.requires_grad_(pos_embed_grad)

        for p in self.model.decoder.embed_positions.parameters():
            p.requires_grad_(pos_embed_grad)

        print(
          """
             |---------------------------------------|
             |    Layer Name         |    Status     |
             |---------------------------------------|
                  Tok Embedding      |   {}       
             |---------------------------------------|
          """.format(m1)
        )
    
    def norm_requires_grad_(self, 
                            enc_norm: bool, 
                            dec_norm: bool,
                            cross_attn_norm: bool):

        # encoder specific
        num = len(self.model.encoder.layers)
        for i in range(num):
            for p in self.model.encoder.layers[i].final_layer_norm.parameters():
                p.requires_grad_(enc_norm)

            for p in self.model.encoder.layers[i].self_attn_layer_norm.parameters():
                p.requires_grad_(enc_norm)

        for p in self.model.encoder.layer_norm.parameters():
            p.requires_grad_(enc_norm)

        for p in self.model.encoder.layernorm_embedding.parameters():
            p.requires_grad_(enc_norm)

        if enc_norm:
            m1 = "trainable"
        else:
            m1 = "freezed"

        # cross attn
        num = len(self.model.decoder.layers)
        for i in range(num):
            for p in self.model.decoder.layers[i].encoder_attn_layer_norm.parameters():
                p.requires_grad_(cross_attn_norm)

        if cross_attn_norm:
            m3 = "trainable"
        else:
            m3 = "freezed"

        # decoder specific
        num = len(self.model.decoder.layers)
        for i in range(num):
            for p in self.model.decoder.layers[i].final_layer_norm.parameters():
                p.requires_grad_(dec_norm)

            for p in self.model.decoder.layers[i].self_attn_layer_norm.parameters():
                p.requires_grad_(dec_norm)

        for p in self.model.decoder.layer_norm.parameters():
            p.requires_grad_(dec_norm)

        for p in self.model.decoder.layernorm_embedding.parameters():
            p.requires_grad_(dec_norm)

        if dec_norm:
            m2 = "trainable"
        else:
            m2 = "freezed"

        print(
          """
             |-----------------------------------|
             |    Layer Name   |     Status      |
             |-----------------------------------|
                Encoder  Norm  |    {}       
             |-----------------------------------|
                Decoder Norm   |    {}       
             |-----------------------------------|
                Enc-Dec Norm   |    {}       
             |-----------------------------------|
          """.format(m1, m2, m3)
        )

    def save_specific_layers(self, path=None, enc_self_attn=False, tok_embed=False, dec_ffn=False):
        state_dict = self.state_dict()

        saving_keys = []

        if dec_ffn:
            num = len(self.model.decoder.layers)
            for i in range(num):
                key = f"model.decoder.layers.{i}.fc"
                saving_keys.extend([k for k in state_dict.keys() if k.startswith(key)])

        if enc_self_attn:
            num = len(self.model.encoder.layers)
            for i in range(num):
                key = f"model.encoder.layers.{i}.self_attn"
                saving_keys.extend([k for k in state_dict.keys() if k.startswith(key)])

        if tok_embed:
            key = "model.shared"
            saving_keys.extend([k for k in state_dict.keys() if k.startswith(key)])

        saving = {}
        for k in saving_keys:
            saving.update({k: state_dict[k]})

        if path:
            print(f"saving: {saving.keys()}")
            torch.save(saving, path)

    def load_specific_layers(self, path: str = None, map_location="cuda:0"):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=False)
        print(f'Loading {state_dict.keys()}')
