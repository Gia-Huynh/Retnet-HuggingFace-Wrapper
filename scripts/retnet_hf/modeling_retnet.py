from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_retnet import RetNetConfig
from .retnet import RetNetDecoder, init_bert_params


@dataclass
class RetNetModelOutputWithPast(BaseModelOutputWithPast):
    aux_loss: torch.FloatTensor | None = None
    aux_losses: tuple[torch.FloatTensor | None, ...] | None = None


class RetNetPreTrainedModel(PreTrainedModel):
    config_class = RetNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RetNetModel):
            module.config.gradient_checkpointing = value
            module.decoder.args.checkpoint_activations = value

    def _init_weights(self, module):
        # We initialize through TorchScale's helper instead of Transformers' default initializer.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            init_bert_params(module)


class RetNetModel(RetNetPreTrainedModel):
    def __init__(self, config: RetNetConfig):
        super().__init__(config)
        args = self._config_to_torchscale_args(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder = RetNetDecoder(args=args, embed_tokens=self.embed_tokens)
        self.post_init()

    @staticmethod
    def _config_to_torchscale_args(config: RetNetConfig) -> SimpleNamespace:
        # This namespace bridges Hugging Face config names to the TorchScale RetNet args API.
        return SimpleNamespace(
            vocab_size=config.vocab_size,
            decoder_embed_dim=config.hidden_size,
            decoder_value_embed_dim=config.value_dim,
            decoder_ffn_embed_dim=config.intermediate_size,
            decoder_layers=config.num_hidden_layers,
            decoder_retention_heads=config.num_retention_heads,
            activation_fn=config.hidden_act,
            dropout=config.hidden_dropout_prob,
            activation_dropout=config.activation_dropout,
            drop_path_rate=config.drop_path_rate,
            recurrent_chunk_size=config.recurrent_chunk_size,
            chunkwise_recurrent=config.chunkwise_recurrent,
            decoder_normalize_before=config.decoder_normalize_before,
            layernorm_eps=config.rms_norm_eps,
            no_scale_embedding=config.no_scale_embedding,
            layernorm_embedding=config.layernorm_embedding,
            share_decoder_input_output_embed=config.tie_word_embeddings,
            deepnorm=config.deepnorm,
            fsdp=config.fsdp,
            checkpoint_activations=config.gradient_checkpointing or config.checkpoint_activations,
            no_output_layer=config.no_output_layer,
            moe_freq=config.moe_freq,
            moe_top1_expert=config.moe_top1_expert,
            moe_expert_count=config.moe_expert_count,
            moe_gating_use_fp32=config.moe_gating_use_fp32,
            moe_eval_capacity_token_fraction=config.moe_eval_capacity_token_fraction,
            use_xmoe=config.use_xmoe,
            moe_second_expert_policy=config.moe_second_expert_policy,
            moe_normalize_gate_prob_before_dropping=config.moe_normalize_gate_prob_before_dropping,
            subln=config.subln,
            xpos_rel_pos=config.xpos_rel_pos,
            multiway=config.multiway,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
        self.decoder.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: dict[str, Any] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        is_first_step: bool | None = False,
        **kwargs,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must pass either input_ids or inputs_embeds.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Pass only one of input_ids or inputs_embeds.")
        if attention_mask is not None:
            # The retained-state masking is handled inside TorchScale RetNet; this wrapper keeps
            # the usual HF signature but does not consume attention_mask.
            kwargs.pop("attention_mask", None)

        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )

        incremental_state = past_key_values
        if use_cache and incremental_state is None:
            #incremental_state = {"is_first_step": True}
            is_first_step = True
        elif incremental_state is not None:
            #incremental_state["is_first_step"] = False
            is_first_step = False

        logits_or_hidden, extra = self.decoder(
            prev_output_tokens=input_ids,
            incremental_state=incremental_state if use_cache else None,
            features_only=True,
            return_all_hiddens=output_hidden_states,
            token_embeddings=inputs_embeds,
            is_first_step = is_first_step,
            **kwargs,
        )

        if use_cache and incremental_state is not None:
            #incremental_state["is_first_step"] = False
            is_first_step = False

        hidden_states = tuple(extra["inner_states"]) if output_hidden_states else None
        output = RetNetModelOutputWithPast(
            last_hidden_state=logits_or_hidden,
            past_key_values=incremental_state if use_cache else None,
            hidden_states=hidden_states,
            attentions=None,
            aux_losses=tuple(extra.get("l_aux", [])) if extra.get("l_aux") is not None else None,
        )
        if not return_dict:
            return tuple(v for v in output.to_tuple() if v is not None)
        return output


class RetNetForCausalLM(RetNetPreTrainedModel, GenerationMixin):
    #_tied_weights_keys = ["model.decoder.output_projection.weight"]
    #_tied_weights_keys = None
    _tied_weights_keys = {
        #"lm_head.weight","model.decoder.output_projection.weight",
        #"model.decoder.embed_tokens.weight","model.embed_tokens.weight",
    "model.decoder.embed_tokens.weight": "model.embed_tokens.weight",
        "model.decoder.output_projection.weight":"model.embed_tokens.weight",
        "lm_head.weight" : "model.decoder.output_projection.weight",
    }
    def __init__(self, config: RetNetConfig):
        super().__init__(config)
        self.model = RetNetModel(config)
        # Reuse the decoder's language modeling head directly.
        self.lm_head = self.model.decoder.output_projection

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)
        if self.config.tie_word_embeddings:
            self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.decoder.output_projection = new_embeddings
        self.lm_head = new_embeddings

    def tie_weights(self):
        super().tie_weights()
        self.lm_head = self.model.decoder.output_projection

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, use_cache=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: dict[str, Any] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits, outputs.past_key_values, outputs.hidden_states)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )
