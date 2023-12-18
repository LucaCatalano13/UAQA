import math
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sized, Tuple, Union, cast

import numpy as np
import torch
from einops import repeat
from torch import nn
from torch.jit import Final
from torch.nn import functional as F

from .utils import default_model_path, device
from datasets.CollectionDataset import BANDS_GROUPS_IDX

class Seq2Seq(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    def forward(
        self,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        latlons: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        month: Union[torch.Tensor, int] = 0,
    ):
        raise NotImplementedError

class FinetuningHead(nn.Module):
    def __init__(self, hidden_size: int, num_outputs: int, regression: bool) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.regression = regression
        self.linear = nn.Linear(hidden_size, num_outputs)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        if (not self.regression) & (self.num_outputs == 1):
            x = torch.sigmoid(x)
        return x

class FineTuningModel(nn.Module):
    encoder: nn.Module

    def forward(
        self,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        latlons: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        month: Union[torch.Tensor, int] = 0,
    ) -> torch.Tensor:
        raise NotImplementedError

def param_groups_lrd(
    model: FineTuningModel, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75
):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(cast(Sized, model.encoder.blocks)) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_rest_finetuning(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())

def get_layer_id_for_rest_finetuning(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if "embed" in name:
        return 0
    elif name.startswith("encoder.blocks"):
        return int(name.split(".")[2]) + 1
    else:
        return num_layers

class Attention(nn.Module):
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")  # FIXME

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    """Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)"""

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).to(device)

def get_day_of_week_encoding_table(d_hid):
    """Sinusoid day of week encoding table, for 7 days indexed from 0-6"""
    assert d_hid % 4 == 0
    angles = np.arange(0, 8) / (7 / (2 * np.pi))
    sin_table = np.sin(np.stack([angles for _ in range(d_hid // 4)], axis=-1))
    cos_table = np.cos(np.stack([angles for _ in range(d_hid // 4)], axis=-1))
    day_of_week_table = np.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)
    return torch.FloatTensor(day_of_week_table).to(device)

def get_day_of_year_encoding_table(d_hid):
    """Sinusoid days of the year encoding table, for 366 days indexed from 0-365"""
    assert d_hid % 4 == 0
    angles = np.arange(0, 367) / (366 / (2 * np.pi))
    sin_table = np.sin(np.stack([angles for _ in range(d_hid // 4)], axis=-1))
    cos_table = np.cos(np.stack([angles for _ in range(d_hid // 4)], axis=-1))
    day_of_year_table = np.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)
    return torch.FloatTensor(day_of_year_table).to(device)

class Encoder(nn.Module):
    def __init__(
        self,
        embedding_size: int = 128,
        channel_embed_ratio: float = 0.25,
        temp_embed_ratio: float = 0.25,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
    ):
        super().__init__()

        self.band_groups = BANDS_GROUPS_IDX
        self.embedding_size = embedding_size

        # this is used for the channel embedding
        self.band_group_to_idx = {
            group_name: idx for idx, (group_name, _) in enumerate(self.band_groups.items())
        }
        # one linear mapping for each channel group
        self.eo_patch_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(len(group), embedding_size)
                for group_name, group in self.band_groups.items()
            }
        )
        # linear mapping for the latlon (3 because x,y,z functions: cos(lat) x cos(lon), cos(lat) x sin(lon), sin(lat))
        self.latlon_embed = nn.Linear(3, embedding_size)
        # the transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_size)

        # the positional + monthly + channel embedding
        self.max_sequence_length = max_sequence_length
        pos_embedding_size = int(embedding_size * (1 - (channel_embed_ratio + temp_embed_ratio)))
        channel_embedding_size = int(embedding_size * channel_embed_ratio)
        temp_embedding_size = int(embedding_size * temp_embed_ratio)
        # the positional embedding whitin the sequence
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_sequence_length, pos_embedding_size), requires_grad=False
        )
        # generate tables for lookup
        # day of year [0-365] in tot 366 days (all leap years)
        day_of_year_tab = get_day_of_year_encoding_table(temp_embedding_size)
        day_of_week_tab = get_day_of_week_encoding_table(temp_embedding_size)
        # lookup tables for temporal emeddings
        self.day_of_year_embed = nn.Embedding.from_pretrained(day_of_year_tab, freeze=True)
        self.day_of_week_embed = nn.Embedding.from_pretrained(day_of_week_tab, freeze=True)
        # TODO: len(self.band_groups) + 1 per latlons?
        self.channel_embed = nn.Embedding(
            num_embeddings=len(self.band_groups), embedding_dim=channel_embedding_size
        )

        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_sinusoid_encoding_table(self.pos_embed.shape[1], self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def cartesian(latlons: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # an embedding is calculated for all timesteps. This is then expanded
            # for each timestep in the sequence
            latlon_radians = latlons * math.pi / 180
            lats, lons = latlon_radians[:, :, 0], latlon_radians[:, :, 1]
            x = torch.cos(lats) * torch.cos(lons)
            y = torch.cos(lats) * torch.sin(lons)
            z = torch.sin(lats)
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def mask_tokens(x, mask):
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]
        # TODO: mask value = 0?
        print("Mask tokens: ", x.isnan().sum())
        x[mask.bool()] = 0
        print("Mask tokens: ", x.isnan().sum())
        x = x.view(batch_size, x.shape[-2], embedding_dim)
        print("Mask tokens: ", x.isnan().sum())
        return x

    def forward(
        self,
        x: torch.Tensor,
        latlons: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        day_of_week: Union[torch.Tensor, int] = 0,
        day_of_year: Union[torch.Tensor, int] = 0,
        eval_task: bool = True,
    ):

        if mask is None:
            mask = torch.zeros_like(x, device=x.device).float()
        # array of the dayOfWeek, dayOfYear analized in the batch
        # ==> given in function arguments
        # encode the months using sin cos --> paper's formula
        day_of_week_embedding = self.day_of_week_embed(day_of_week)
        day_of_year_embedding = self.day_of_year_embed(day_of_year)

        positional_embedding = repeat(
            self.pos_embed[:, : x.shape[1], :], "b t d -> (repeat b) t d", repeat=x.shape[0]
        )

        # we assume the number of masked patches is the same
        # for all items in the batch. Otherwise things become a headache
        all_tokens, all_masks = [], []
        print("X nan: ", x.isnan().sum())
        # iterate over the different channel groups (name of the dataset) and chanel_idxs (indexes of the bands in BANDS (enum))
        for channel_group, channel_idxs in self.band_groups.items():
            # for each channel group, create a dictionary that has as keys the name of the datasets (channel_group) 
            # and as values a linear combination (FC) from the number of bands in the channel_group
            # to a space of dimension(embedding_size)
            # return an initial embedding of the channel group
            tokens = self.eo_patch_embed[channel_group](x[:, :, channel_idxs])
            # create an embedding of the channel group --> lookup table
            channel_embedding = self.channel_embed(
                torch.tensor(self.band_group_to_idx[channel_group]).long().to(device)
            )
            channel_embedding = repeat(channel_embedding, "d -> b t d", b=x.shape[0], t=x.shape[1])
            
            # create the embedding of the channel group
            # Encoding of the channel group
            channel_wise_positional_embedding = torch.cat(
                (day_of_year_embedding, day_of_week_embedding, channel_embedding, positional_embedding), dim=-1
            )
            indices = slice(None) # it talkes all the elements of the tensor

            tokens = tokens[:, indices]
            # add the positional embedding to the tokens (initial token embedding)
            tokens += channel_wise_positional_embedding
            # all tokens has one element (that is tokens) for each channel group
            all_tokens.append(tokens)
            group_mask = repeat(
                torch.max(mask[:, indices, channel_idxs], dim=-1)[0],
                "b t -> b t d",
                d=tokens.shape[-1],
            )
            all_masks.append(group_mask)
        print("X nan: ", x.isnan().sum())   
        # TODO: separate timesteps and channels? --> probably origin of 1 in dimensions
        x = torch.cat(all_tokens, dim=1)
        print("X nan: ", x.isnan().sum())   
        mask = torch.cat(all_masks, dim=1)
        print("X nan: ", x.isnan().sum())   
        x = self.mask_tokens(x, mask)
        print("X nan: ", x.isnan().sum())   
        # latlons (BS, timesteps, 2)
        # latlons_cartesian (BS, timesteps, 3)
        # latlons_token (BS, 1, 128)
        latlon_tokens = self.latlon_embed(self.cartesian(latlons)[:, 0, :]).unsqueeze(1)
        print("X nan: ", x.isnan().sum())   
        # append lat_lon token to the embedding
        # x (BS, len(band_idx)*timesteps, 128) --> (BS, (len(band_idx)*timesteps)+1, 128)
        # lend(band_ix) == n_channel_groups (in our case == n_datasets becouse we do not create subgroup of datasets)
        x = torch.cat((latlon_tokens, x), dim=1)
        print("X nan after mask: ", x.isnan().sum())
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if eval_task:
            return self.norm(x.mean(dim=1))
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(
        self,
        channel_embeddings: nn.Embedding,
        encoder_embed_dim=128,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=8,
        mlp_ratio=2,
        max_sequence_length=24,
    ):
        super().__init__()

        self.band_groups = BANDS_GROUPS_IDX

        # this is used for the channel embedding
        self.band_group_to_idx = {
            group_name: idx for idx, (group_name, _) in enumerate(self.band_groups.items())
        }

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.eo_decoder_pred = nn.ModuleDict(
            {
                group_name: nn.Linear(decoder_embed_dim, len(group))
                for group_name, group in self.band_groups.items()
            }
        )

        self.channel_embeddings = channel_embeddings
        channel_embedding_dims = channel_embeddings.weight.shape[-1]
        # TODO: cosa significa??
        remaining_embeddings = decoder_embed_dim - channel_embedding_dims
        # the positional + monthly + channel embedding
        self.max_sequence_length = max_sequence_length
        # internally decoder works with dimension reminaining_embeddings//2
        # TODO: why?
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_sequence_length, int(remaining_embeddings) // 2),
            requires_grad=False,
        )
        # generate tables for lookup
        day_of_year_tab = get_day_of_year_encoding_table(int(remaining_embeddings) // 2)
        day_of_week_tab = get_day_of_week_encoding_table(int(remaining_embeddings) // 2)
        
        # lookup tables for temporal emeddings
        self.day_of_year_embed = nn.Embedding.from_pretrained(day_of_year_tab, freeze=True)
        self.day_of_week_embed = nn.Embedding.from_pretrained(day_of_week_tab, freeze=True)
        
        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_sinusoid_encoding_table(self.pos_embed.shape[1], self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def add_embeddings(self, x, day_of_week: Union[torch.Tensor, int], day_of_year: Union[torch.Tensor, int]):
        num_channel_groups = len(self.band_group_to_idx)
        # -1 since we remove latlon token (one per batch element (cut of a pixel timeseries))
        num_timesteps = int((x.shape[1] - 1) / (num_channel_groups))
        # day_of_week (BS, timesteps) --> (BS, timesteps*len(band_idx), 128/4)
        day_of_week_embedding = repeat(
            self.day_of_week_embed(day_of_week), "b t d -> b (repeat t) d", repeat=num_channel_groups
        )
        # day_of_year (BS, timesteps) --> (BS, timesteps*len(band_idx), 128/4)
        day_of_year_embedding = repeat(
            self.day_of_year_embed(day_of_year), "b t d -> b (repeat t) d", repeat=num_channel_groups
        )
        # positional_embedding (BS, timesteps*len(band_idx), 128/4)
        positional_embedding = repeat(
            self.pos_embed[:, :num_timesteps, :],
            "b t d -> (b2 b) (t2 t) d",
            b2=x.shape[0],
            t2=num_channel_groups,
        )
        # TODO: self.channel_embeddings.weight is the matrix of the embeddings of the channel groups in the encoder. Why?
        # channel_embeddings (timesteps*len(band_idx), 128/4)
        channel_embeddings = torch.repeat_interleave(
            self.channel_embeddings.weight, repeats=num_timesteps, dim=0
        )
        # channel_embeddings (BS, timesteps*len(band_idx), 128/4)
        channel_embeddings = repeat(channel_embeddings, "c d -> b c d", b=x.shape[0])
        # positional_embedding (BS, timesteps*len(band_idx), 128) 128 = sum of the fraction of the embedding size above
        positional_embedding = torch.cat(
            (day_of_year_embedding, day_of_week_embedding, channel_embeddings, positional_embedding), dim=-1
        )
        # add the zero embedding for the latlon token
        # positional_embedding (BS, timesteps*len(band_idx)+1, 128)
        positional_embedding = torch.cat(
            [torch.zeros_like(positional_embedding[:, 0:1, :]), positional_embedding], dim=1
        )
        x += positional_embedding
        return x

    def reconstruct_inputs(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # remove the latlon token
        x = x[:, 1:, :]
        # split into channel groups
        num_channel_groups = len(self.band_group_to_idx)
        num_timesteps = int((x.shape[1]) / num_channel_groups)

        #TODO: shape dimension = 4?
        x = x.view(x.shape[0], num_channel_groups, num_timesteps, x.shape[-1])

        eo_output = []
        for group_name, idx in self.band_group_to_idx.items():
            group_tokens = x[:, idx]
            eo_output.append(self.eo_decoder_pred[group_name](group_tokens))

        # we can just do this concatenation because the BANDS_GROUP_IDX
        # is ordered
        return torch.cat(eo_output, dim=-1)

    def forward(self, x, day_of_week, day_of_year):
        # FC
        # (BS, 27, 128) --> 27 perchè 25 (ntimesteps*n_bands_idx) + 2 (latlon)
        x = self.decoder_embed(x)
        x = self.add_embeddings(x, day_of_week, day_of_year)
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.reconstruct_inputs(x)

class PrestoFineTuningModel(FineTuningModel):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder: Encoder = deepcopy(encoder)
        # make sure the model is trainable, since we can call
        # this having called requires_grad_(False)
        self.encoder.requires_grad_(True)
        # but don't unfreeze the position encoder, which
        # shouldn't be trainable
        self.encoder.pos_embed.requires_grad_(False)
        self.encoder.month_embed.requires_grad_(False)
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        latlons: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        day_of_week: Union[torch.Tensor, int] = 0,
        day_of_year: Union[torch.Tensor, int] = 0,
    ) -> torch.Tensor:

        return self.head(
            self.encoder(
                x=x,
                latlons=latlons,
                mask=mask,
                day_of_year=day_of_year,
                day_of_week=day_of_week,
                eval_task=True,
            )
        )

class Presto(Seq2Seq):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        latlons: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        day_of_year: Union[torch.Tensor, int] = 0,
        day_of_week: Union[torch.Tensor, int] = 0
    ) -> torch.Tensor:
        print("x nan:" , x.isnan().sum())
        encoded_x = self.encoder(
            x=x,
            latlons=latlons,
            mask=mask,
            day_of_year=day_of_year,
            day_of_week=day_of_week,
            eval_task=False,
        )
        print(encoded_x.isnan().sum())
        reconstructed_x = self.decoder(encoded_x, day_of_week, day_of_year)
        print(reconstructed_x.isnan().sum())
        return reconstructed_x

    @classmethod
    def construct(
        cls,
        encoder_embedding_size: int = 128,
        channel_embed_ratio: float = 0.25,
        month_embed_ratio: float = 0.25,
        encoder_depth=2,
        mlp_ratio=4,
        encoder_num_heads=8,
        decoder_embedding_size=128,
        decoder_depth=2,
        decoder_num_heads=8,
        max_sequence_length=24,
    ):
        encoder = Encoder(
            embedding_size=encoder_embedding_size,
            channel_embed_ratio=channel_embed_ratio,
            month_embed_ratio=month_embed_ratio,
            depth=encoder_depth,
            mlp_ratio=mlp_ratio,
            num_heads=encoder_num_heads,
            max_sequence_length=max_sequence_length,
        )
        decoder = Decoder(
            channel_embeddings=encoder.channel_embed,
            encoder_embed_dim=encoder_embedding_size,
            decoder_embed_dim=decoder_embedding_size,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
        )
        return cls(encoder, decoder)

    def construct_finetuning_model(
        self,
        num_outputs: int,
        regression: bool = False,
    ):
        head = FinetuningHead(
            num_outputs=num_outputs,
            hidden_size=self.encoder.embedding_size,
            regression=regression,
        )
        model = PrestoFineTuningModel(self.encoder, head).to(device)
        model.train()
        return model

    @classmethod
    def load_pretrained(cls, model_path: Union[str, Path] = default_model_path):
        model = cls.construct()
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
