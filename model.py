'''
Adapted from https://github.com/huggingface/transformers
'''

from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, __HEAD_MASK_WARNING_MSG
import copy
import os
import warnings
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)

import numpy as np
import os

def flatten_and_save_to_txt(array, file_path):
    array = array.cpu().numpy()

    bs = array.shape[0]
    flattened_array = array.reshape(bs, -1)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'a') as file:
        np.savetxt(file, flattened_array, fmt='%f')

    print(f"张量已保存到文件：{file_path}")


def info_nce_loss(text_sem, image_sem, temperature=0.07):
    bs, seq_length = text_sem.size(0), text_sem.size(1)
    text_sem = text_sem.view(bs, -1)    # bs x -1
    image_sem = image_sem.view(bs, -1)  # bs x -1

    similarity_scores = torch.matmul(text_sem, image_sem.T) 
    logits = F.softmax(similarity_scores / temperature, dim=1)  

    exp_neg_sum = torch.sum(logits, dim=0)  # [bs]
    exp_pos = torch.diagonal(logits)        # [bs]
    losses = exp_pos / exp_neg_sum  # [bs]
    loss = torch.sum(losses, dim=0) 

    return loss


def l2_loss(A, B):
    assert A.shape == B.shape, "输入数组A和B的形状必须相同"
    loss = np.sum((A - B) ** 2)
    
    return loss


class MaxMeanDifferenceCalculator(nn.Module):
    def __init__(self):
        super(MaxMeanDifferenceCalculator, self).__init__()

    def forward(self, last_element_a, last_element_b):
        if not isinstance(last_element_a, torch.Tensor) or not isinstance(last_element_b, torch.Tensor):
            raise ValueError("Both inputs must be PyTorch tensors.")
        differences = last_element_a - last_element_b
        abs_differences = torch.abs(differences)
        mean_difference = torch.mean(abs_differences, dim=-1)
        max_mean_difference = torch.max(mean_difference)
        return max_mean_difference

class JensenShannonDivergence:
    def __init__(self):
        pass

    def kl_divergence(self, p, q):
        return F.kl_div(F.log_softmax(p, dim=1), F.softmax(q, dim=1), reduction='batchmean')

    def js_divergence(self, logit_a, logit_b):
        prob_a = F.softmax(logit_a, dim=1)
        prob_b = F.softmax(logit_b, dim=1)
        m = (prob_a + prob_b) / 2
        jsd = self.kl_divergence(prob_a.log(), m) + self.kl_divergence(prob_b.log(), m)
        return jsd / 2

class KLDivergenceCalculator:
    def __init__(self):
        pass

    def calculate_kl_divergence(self, logit_a, logit_b):
        prob_a = F.softmax(logit_a, dim=1)
        prob_b = F.softmax(logit_b, dim=1)  # 目标分布
        kl_div = F.kl_div(F.log_softmax(logit_a, dim=1), prob_b, reduction='batchmean')
        return kl_div
    
class CrossModalAttention(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super(CrossModalAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5
        self.query_proj = nn.Linear(emb_dim, emb_dim)
        self.key_proj = nn.Linear(emb_dim, emb_dim)
        self.value_proj = nn.Linear(emb_dim, emb_dim)
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, heat, text_visual_heat):
        bs, seq_len_text, emb_dim = heat.size()    
        _, seq_len_visual, _ = text_visual_heat.size()
        
        Q = self.query_proj(heat)  # [bs, 512, 768]
        K = self.key_proj(text_visual_heat)  # [bs, 612, 768]
        V = self.value_proj(text_visual_heat)  # [bs, 612, 768]
        
        attention_weights = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        context_vector = torch.matmul(attention_weights, V)  # [bs, 512, 768]

        heat_enhanced = heat + context_vector
        normalized_states = self.layer_norm(heat_enhanced)
        return heat_enhanced
    

class MCIM(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(MCIM, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
    
        self.conv1d_t = nn.Conv1d(768, 768, kernel_size=1)
        self.conv1d_v = nn.Conv1d(768, 768, kernel_size=1)
        self.ca_t = CrossModalAttention(in_channels=768, emb_dim=768)
        self.ca_v = CrossModalAttention(in_channels=768, emb_dim=768)
        self.ca = CrossModalAttention(in_channels=768, emb_dim=768)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)

    def forward(self, text, visual):
        text_heat = text
        visual_heat = visual
        text_heat = self.conv1d_t(text_heat.transpose(1, 2))        # [bs, 768, 512]
        visual_heat = self.conv1d_v(visual_heat.transpose(1, 2))    # [bs, 768, 100]
        text_heat = text_heat.transpose(1, 2)                       # [bs, 512, 768]
        visual_heat = visual_heat.transpose(1, 2)                   # [bs, 100, 768]
        text_visual_embedding = torch.cat((text_heat, visual_heat), dim=1)  # [bs, 612, 768]

        text_heat = self.ca_t(text_heat, text_visual_embedding)         # [bs, 512, 768]
        visual_heat = self.ca_v(visual_heat, text_visual_embedding)     # [bs, 100, 768]

        text_heat = self.ca(text_heat, visual_heat)         # [bs, 512, 768]
        output = self.transformer_encoder(text_heat)   # [bs, 612, 768]
        return output


class T5ForMultimodalGenerationMCCoT(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, patch_size, padding_idx, save_dir, vot_num,alpha):
        super().__init__(config)
        self.model_dim = config.d_model # 768
        self.vot_num = vot_num   # 3
        self.alpha=alpha    # 0.5
        self.padding_idx = padding_idx  # 0

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_num, self.patch_dim = patch_size# 100, 256

        self.image_dense = nn.Linear(self.patch_dim, config.d_model)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size, vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.gate_dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        # self.encoder = JointEncoder(encoder_config, self.shared, patch_size)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(2*config.d_model, config.vocab_size, bias=False)

        # ours
        self.MCIM = MCIM(d_model=768, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1)
        self.fc_for_logit = nn.Linear(config.d_model, 2, bias=False)
        self.fc_dim = nn.Linear(612, 512, bias=False)

        # -------------------------------------------------------------------
        self.post_init()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,   # zg
        input_ids_masked: Optional[torch.LongTensor] = None,   # zg
        image_ids=None,     # zg
        attention_mask: Optional[torch.FloatTensor] = None, # zg
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,  # zg
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        CoT_ids: Optional[torch.LongTensor] = None, # zg
        CoT_attention_mask: Optional[torch.FloatTensor] = None, # zg
        subject=None,     # zg      [8]
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache               # True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict   # True

        if head_mask is not None and decoder_head_mask is None:     # NO
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if encoder_outputs is None: # True
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=True,  # 设置这里 output_attentions
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            encoder_outputs_masked = self.encoder(
                input_ids=input_ids_masked,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=True,  # 设置这里 output_attentions
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):      # NO
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            encoder_outputs_masked = BaseModelOutput(
                last_hidden_state=encoder_outputs_masked[0],
                hidden_states=encoder_outputs_masked[1] if len(encoder_outputs_masked) > 1 else None,
                attentions=encoder_outputs_masked[2] if len(encoder_outputs_masked) > 2 else None,
            )
        
        hidden_states = encoder_outputs[0]                  # 文本 [bs, 512, 768]
        hidden_states_masked = encoder_outputs_masked[0]    # 被随机mask的文本 [bs, 512, 768]
        image_embedding = self.image_dense(image_ids)       # [bs, 100, 768]

        output_Trans_En_a = self.MCIM(hidden_states, image_embedding)   # [bs, 512, 768]
        output_Trans_En_b = self.MCIM(hidden_states_masked, image_embedding)   # [bs, 512, 768]

        last_element_a = output_Trans_En_a[:, -1, :]    # [bs, 768]
        last_element_b = output_Trans_En_b[:, -1, :]    # [bs, 768]
        logit_a = self.fc_for_logit(last_element_a)     # [bs, 2]
        logit_b = self.fc_for_logit(last_element_b)     # [bs, 2]

        image_embedding = self.image_dense(image_ids)
        image_att, _ = self.mha_layer(hidden_states, image_embedding, image_embedding)
        merge = torch.cat([hidden_states, image_att], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        hidden_states = (1 - gate) * hidden_states + gate * image_att

        if self.model_parallel: # False
            torch.cuda.set_device(self.decoder.first_device)
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:  # True
            decoder_input_ids = self._shift_right(labels)# [bs, 64]

        if self.model_parallel: # False
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        all_logits = []
        for i in range(3):   # 5
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = decoder_outputs[0]
            decoder_outputs_a = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=output_Trans_En_a,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output_a = decoder_outputs_a[0]
            del decoder_outputs_a
            decoder_outputs_b = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=output_Trans_En_b,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output_b = decoder_outputs_b[0]
            del decoder_outputs_b

            # Set device for model parallelism
            if self.model_parallel:     # False
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                sequence_output = sequence_output.to(self.lm_head.weight.device)
                sequence_output_a = sequence_output_a.to(self.lm_head.weight.device)
                sequence_output_b = sequence_output_b.to(self.lm_head.weight.device)
            if self.config.tie_word_embeddings:     # True
                sequence_output = sequence_output * (self.model_dim**-0.5)
                sequence_output_a = sequence_output_a * (self.model_dim**-0.5)
                sequence_output_b = sequence_output_b * (self.model_dim**-0.5)

            lm_logits = self.lm_head(sequence_output)
            lm_logits_a = self.lm_head(sequence_output_a)
            lm_logits_b = self.lm_head(sequence_output_b)
            all_logits.append(lm_logits)
            all_logits.append(lm_logits_a)
            all_logits.append(lm_logits_b)
            del sequence_output, sequence_output_a, sequence_output_b, lm_logits, lm_logits_a, lm_logits_b
            
        
        # voting
        stacked_logits = torch.stack(all_logits, dim=0) 
        mean_logits = torch.mean(stacked_logits, dim=0)
        stddev_logits = torch.std(stacked_logits, dim=0)
        weights = 1 / (1 + stddev_logits) 
        weighted_mean_logits = torch.sum(weights * stacked_logits, dim=0) / torch.sum(weights, dim=0)
        alpha = self.alpha
        lm_logits = alpha * mean_logits + (1 - alpha) * weighted_mean_logits
        
        loss = None
        if labels is not None:
            loss_CEL = CrossEntropyLoss(ignore_index=-100)
            mmd_loss = MaxMeanDifferenceCalculator()
            jsd = JensenShannonDivergence()
            kl_calculator = KLDivergenceCalculator()

            mmd = mmd_loss(last_element_a, last_element_b)
            js_div = jsd.js_divergence(logit_a, logit_b)
            kl_div = kl_calculator.calculate_kl_divergence(logit_a, logit_b) 

            loss_inf = loss_CEL(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            alpha_inf, alpha_mmd, alpha_js_kl, alpha_kl_USD = 0.5, 1, 1, 0
            loss = alpha_inf * loss_inf + alpha_mmd * mmd + alpha_js_kl * js_div
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

