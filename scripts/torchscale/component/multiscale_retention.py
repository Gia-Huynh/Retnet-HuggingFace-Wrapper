# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]


import torch
import torch.nn.functional as F
from torch import nn
from .rms_norm import RMSNorm

from .multiway_network import MultiwayWrapper

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError
    
class MultiScaleRetention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        value_dim,
        num_heads,
        gate_fn="swish",
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = self.value_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        
        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=False))
        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=False))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, value_dim, bias=False))
        self.g_proj = MultiwayWrapper(args, nn.Linear(embed_dim, value_dim, bias=False))
        
        self.out_proj = MultiwayWrapper(args, nn.Linear(value_dim, embed_dim, bias=False))

        self.group_norm = MultiwayWrapper(args, RMSNorm(self.head_dim, eps=args.layernorm_eps, elementwise_affine=False))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2) # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output

    def recurrent_forward_backup(
        self,
        qr, kr, v,
        decay,
        incremental_state,
        idx
    ):
        bsz = v.size(0)

        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v
        
        #Check if first run
        try:
            layer = incremental_state.layers[idx]
            temp_first_run = False
            if layer.keys is None or layer.values is None:
                temp_first_run = True
        except Exception as e:
            temp_first_run = True
            
        if temp_first_run == False:
            prev_kv, prev_scale = incremental_state.layers[idx].keys, incremental_state.layers[idx].values.squeeze()
            if len(prev_scale.shape) > 1:
                prev_scale = prev_scale[:,-1]
            prev_kv = prev_kv.permute(dims=[0,2,1,3])[:,-prev_scale.shape[0]:,:,:]
            scale = prev_scale * decay + 1
            kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(self.num_heads, 1, 1) + kv / scale.sqrt().view(self.num_heads, 1, 1)
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        else:
            scale = torch.ones_like(decay) #If first run

        scale_incremental_vector = scale.view(1, -1, 1, 1)
        kv_incremental_vector = kv.permute(dims=[0,2,1,3])
        incremental_state.update(key_states = kv_incremental_vector, value_states = scale_incremental_vector, layer_idx = idx)
        output = torch.sum(qr * kv, dim=3)
        return output
    
    def recurrent_forward(
        self,
        qr, kr, v,
        decay,
        incremental_state,
        idx
    ):
        print ("Beginning of run")
        print ("decay shape should be (num_head, 1, 1):", decay.shape)
        bsz = v.size(0) #Batchsize
        v = v.view(kr.shape) #bsz, self.num_heads, self.head_dim, v.shape[1])
        print ("Sanity check: q, k, v, their shape should be (bsz, num_head, len, qkv_dim):", qr.shape, kr.shape, v.shape)
        
        kv = k.unsqueez(-1) * v.unsqueeze(-2)
        print ("kv.shape: ", kv.shape)
        
        #Check if first run
        try:
            layer = incremental_state.layers[idx]
            temp_first_run = False
            if layer.keys is None or layer.values is None:
                temp_first_run = True
        except Exception as e:
            temp_first_run = True
            
        if (incremental_state is None) 
            or (len(past_key_values.layers )==0)
            or (all(not layer.is_initialized for layer in past_key_values.layers)):
                temp_first_run = True
        else:
            
                temp_first_run = False
            
        if temp_first_run == False:
            prev_kv, prev_scale = incremental_state.layers[idx].keys, incremental_state.layers[idx].values
            #prev_scale = prev_scale.squeeze()
            if len(prev_scale.shape) > 1:
                #prev_scale = prev_scale[:,-1]
                print ("prev_scale extracted shape:", prev_scale.shape)
                prev_scale = prev_scale[0,:,0,0].squeeze()
                print ("prev_scale after modification's shape (should be [8]):", prev_scale.shape)
            print ("prev_kv extracted shape:", prev_kv.shape)
            prev_kv = prev_kv[:,:,-prev_kv.shape[-1]:,:] #Dynamic Cache ([1, 8, 256, 128]) to ([1, 8, 128, 128]), no-op if StaticCache
            #prev_kv = prev_kv.permute(dims=[0,2,1,3])[:,-prev_scale.shape[0]:,:,:]
            scale = prev_scale * decay + 1
            kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(self.num_heads, 1, 1) + kv / scale.sqrt().view(self.num_heads, 1, 1)
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        else:
            scale = torch.ones_like(decay) #If first run

        print ("scale.shape: ", scale.shape)
        scale_padded = torch.clone(scale[None,:,None,None].repeat(1,1,kv.shape[2],kv.shape[3])) #.repeat(kv.shape[2], axis = 2).repeat(kv.shape[3], axis = 3)
        print ("scale_padded shape (Result): ", scale_padded.shape)
        kv_incremental_vector = kv #.permute(dims=[0,2,1,3])
        print ("kv_incremental_vector (Result): ", kv_incremental_vector.shape)
        incremental_state.update(key_states = kv_incremental_vector, value_states = scale_padded, layer_idx = idx)
        output = torch.sum(qr * kv, dim=3)
        print ("cache.keys.shape: ", incremental_state.layers[idx].keys.shape)
        print ("cache.values.shape: ", incremental_state.layers[idx].values.shape)
        print ("Updated sucessfully")
        print ("Output Type:", type(output))
        print ("Output Shape:", output.shape)
        #if temp_first_run == False:
        #    STOP
        return output
    def chunk_recurrent_forward(
        self,
        qr, kr, v,
        inner_mask
    ):
        mask, cross_decay, query_inner_decay, value_inner_decay = inner_mask
        bsz, tgt_len, embed_dim = v.size()
        chunk_len = mask.size(1)
        num_chunks = tgt_len // chunk_len

        assert tgt_len % chunk_len == 0

        qr = qr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        kr = kr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(2, 3)

        kr_t = kr.transpose(-1, -2)

        qk_mat = qr @ kr_t # bsz * num_heads * chunk_len * chunk_len
        qk_mat = qk_mat * mask
        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mat = qk_mat / inner_scale
        inner_output = torch.matmul(qk_mat, v) # bsz * num_heads * num_value_heads * chunk_len * head_dim
        
        # reduce kv in one chunk
        kv = kr_t @ (v * value_inner_decay)

        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)
        
        # accumulate kv by loop
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).max(dim=-1, keepdim=True).values.clamp(min=1)
            
        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)
        
        all_scale = torch.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = (qr * query_inner_decay) @ kv_recurrent
        output = inner_output / align_inner_scale + cross_output / align_cross_scale
        # output = inner_output / cross_scale + cross_output / inner_scale

        output = output.transpose(2, 3)
        return output
    
    def forward(
        self,
        x,
        rel_pos,
        chunkwise_recurrent=False,
        incremental_state=None,
        idx=None
    ):
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if incremental_state is not None: #Could exist but be empty, for example.
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state, idx)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask)
        
        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        output = self.gate_fn(g) * output

        output = self.out_proj(output)

        return output

        
