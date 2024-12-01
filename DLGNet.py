import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
import os

from prediction_module import MLP, LG_BLOCK,SUM_Module,GROUP_Module



import numpy as np
import torch
import torch.nn as nn
import pywt
from numba import jit


class WaveletTransform(nn.Module):
    def __init__(self, L, level, method) -> None:
        super().__init__()
        wavelet = pywt.Wavelet(method)
        self.level = level
        h = wavelet.dec_hi
        g = wavelet.dec_lo
        h_t = torch.from_numpy(np.array(h) / np.sqrt(2))
        g_t = torch.from_numpy(np.array(g) / np.sqrt(2))
        self.w_dec_filter = self.wavelet_dec_filter(h_t, L)
        self.v_dec_filter = self.wavelet_dec_filter(g_t, L)
        self.w_rec_filter = self.wavelet_rec_filter(h_t, L)
        self.v_rec_filter = self.wavelet_rec_filter(g_t, L)
        
    def wavelet_dec_filter(self, wavelet_vec, L):
        
        wavelet_len = wavelet_vec.shape[0]
        filter = torch.zeros(self.level, L, L)
        wl = torch.arange(wavelet_len)
        for j in range(self.level):
            for t in range(L):
                index = torch.remainder(t - 2 ** j * wl, L)
                hl = torch.zeros(L)
                for i, idx in enumerate(index):
                    hl[idx] = wavelet_vec[i]
                filter[j][t] = hl
        return filter   # (level, L, L)


    def wavelet_rec_filter(self, wavelet_vec, L):
        
        wavelet_len = wavelet_vec.shape[0]
        filter = torch.zeros(self.level, L, L)
        wl = torch.arange(wavelet_len)
        for j in range(self.level):
            for t in range(L):
                index = torch.remainder(t + 2 ** j * wl, L)
                hl = torch.zeros(L)
                for i, idx in enumerate(index):
                    hl[idx] = wavelet_vec[i]
                filter[j][t] = hl
        return filter   # (level, L, L)
    

    def modwt(self, x):
        '''
        x: (batch, length, D)
        filters: 'db1', 'db2', 'haar', ...
        '''
        B, L, D = x.shape
        x = x.permute(0, 2, 1)
        w_dec_filter = self.w_dec_filter.to(x)
        v_dec_filter = self.v_dec_filter.to(x)
        v_j = x
        v = []
        for j in range(self.level):
            v_j = torch.einsum('ml,bdl->bdm', v_dec_filter[j], v_j)
            v.append(v_j)
        v = torch.stack(v, dim=2)   # (B, D, level, L)
        v_prime = torch.cat([x.reshape(B, D, 1, L), v[..., :-1, :]], dim=2)  # (B, D, level, L)
        w = torch.einsum('jml,jbdl->bdjm', w_dec_filter, v_prime.permute(2, 0, 1, 3))
        wavecoeff = torch.cat([w, v[..., -1, :].reshape(B, D, 1, L)], dim=2)  # (B, D, level + 1, L)
        
        return wavecoeff.permute(0, 1, 3, 2)  # (B, D, L, level + 1)


    def imodwt(self, wave):
        '''
        wave: (batch, D, length, level + 1)
        '''
        wave = wave.permute(0, 1, 3, 2)
        w_rec_filter = self.w_rec_filter.to(wave)      # (level, L, L)
        v_rec_filter = self.v_rec_filter.to(wave)      # (level, L, L)
        w = wave[..., :-1, :]                               # (B, D, level, L)
        v_j = wave[..., -1, :]          # (B, D, L)
        scale_j = torch.einsum('ml,bdl->bdm', v_rec_filter[-1], v_j).unsqueeze(2)  # (B, D, 1, L)
        for j in range(self.level)[::-1]:
            detail_j = torch.einsum('ml,bdrl->bdrm', w_rec_filter[j], w[..., j, :].unsqueeze(2))
            scale_cat = torch.cat([detail_j, scale_j], dim=2)
            scale_j = torch.einsum('bdrl->bdl', scale_cat)
            if j > 0:
                scale_j = torch.einsum('ml,bdl->bdm', v_rec_filter[j - 1], scale_j).unsqueeze(2)  # (B, D, 1, L)
        
        return scale_j     # (B, D, L)
    
    
    def modwtmra(self, wave):
        ''' Multiresolution analysis based on MODWT'''
        '''
        wave: (batch, D, length, level + 1)
        '''
        wave = wave.permute(0, 1, 3, 2)
        w_rec_filter = self.w_rec_filter.to(wave)      # (level, L, L)
        v_rec_filter = self.v_rec_filter.to(wave)      # (level, L, L)
        w = wave[..., :-1, :]                               # (B, D, level, L)
        v_j = wave[..., -1, :]          # (B, D, L)
        scale_j = torch.einsum('ml,bdl->bdm', v_rec_filter[-1], v_j).unsqueeze(0)
        detail_j = torch.einsum('ml,nbdl->nbdm', w_rec_filter[-1], w[..., -1, :].unsqueeze(0))
        scale_j = torch.cat([detail_j, scale_j], dim=0)
        for j in range(self.level - 1)[::-1]:
            detail_j = torch.einsum('ml,nbdl->nbdm', w_rec_filter[j], w[..., j, :].unsqueeze(0))
            scale_j = torch.einsum('ml,nbdl->nbdm', v_rec_filter[j], scale_j)
            scale_j = torch.cat([detail_j, scale_j], dim=0)
        
        mra = scale_j.permute(1, 2, 0, 3)  # (B, D, level + 1, L)
        recon = torch.einsum('bdjl->bdl', mra)
        return mra.permute(0, 1, 3, 2), recon     # (B, D, L, level + 1)  (B, D, L)



class DLGBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.mra_list = []
        self.number = 0
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
        self.MLP = MLP(self.configs)

        self.lo_go_block = LG_BLOCK(self.configs)

        self.adasum = SUM_Module(self.configs)

        self.group_feature = GROUP_Module(self.configs)

    def forward(self, x):
        # self.number += 1
        B, T, N = x.size()
        
        level = 6
        trans = WaveletTransform(T, level, 'db10')
        w = trans.modwt(x)

        w = self.MLP(w)
        mra, _ = trans.modwtmra(w)

        with torch.no_grad():
            pred_len = self.configs.pred_len
            x_out = x.detach().cpu().numpy()
            mra_out = mra.detach().cpu().numpy()
            mra_path = 'D:/Code/Water_level_predictipon/enc_out/'
            if not os.path.exists(mra_path):
                os.makedirs(mra_path)
            np.save(mra_path + str(self.number) + '_' + str(pred_len) + '_'+    'mra.npy', mra_out)
            np.save(mra_path + str(self.number) + '_' + str(pred_len) + '_'+     'x.npy', x_out)

        #local_global_block
        local_global = self.lo_go_block(mra)


        out = self.conv(mra)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = self.adasum(out,mra)
        res = out + x

        #group feature aggra
        out = self.group_feature(res,self.configs)
        return res


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([DLGBlock(configs)
                                   for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)
        self.number = 0

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self.number += 1
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C] torch.Size([32, 96, 512])
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension # torch.Size([32, 102, 512])
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
            with torch.no_grad():
                enc_out_model = enc_out.detach().cpu().numpy()
                mra_path = 'D:/Code/Water_level_predictipon/enc_out/'
                if not os.path.exists(mra_path):
                    os.makedirs(mra_path)
                np.save(mra_path + str(self.number) + '_' + str(self.pred_len) + '_' + 'block' + str(i) + '.npy', enc_out_model)
        # porject back
        dec_out = self.projection(enc_out) # torch.Size([32, 102, 8]) B,T,D,时间不变，进行维度上的映射 d_model -> c_out

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


