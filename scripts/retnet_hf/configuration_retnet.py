from transformers import PretrainedConfig


class RetNetConfig(PretrainedConfig):
    model_type = "retnet"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 8,
        num_retention_heads: int = 8,
        value_dim: int | None = None,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        activation_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        recurrent_chunk_size: int = 128,
        chunkwise_recurrent: bool = False,
        decoder_normalize_before: bool = True,
        no_scale_embedding: bool = False,
        layernorm_embedding: bool = False,
        tie_word_embeddings: bool = True,
        deepnorm: bool = False,
        rms_norm_eps: float = 1e-6,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 50256,
        eos_token_id: int | None = 50256,
        use_cache: bool = True,
        gradient_checkpointing: bool = False,
        moe_freq: int = 0,
        moe_top1_expert: bool = True,
        moe_expert_count: int = 0,
        moe_gating_use_fp32: bool = True,
        moe_eval_capacity_token_fraction: float = 0.25,
        use_xmoe: bool = False,
        moe_second_expert_policy: str = "random",
        moe_normalize_gate_prob_before_dropping: bool = False,
        subln: bool = False,
        xpos_rel_pos: bool = False,
        multiway: bool = False,
        fsdp: bool = False,
        checkpoint_activations: bool = False,
        no_output_layer: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_retention_heads = num_retention_heads
        self.value_dim = value_dim if value_dim is not None else hidden_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.activation_dropout = activation_dropout
        self.drop_path_rate = drop_path_rate
        self.max_position_embeddings = max_position_embeddings
        self.recurrent_chunk_size = recurrent_chunk_size
        self.chunkwise_recurrent = chunkwise_recurrent
        self.decoder_normalize_before = decoder_normalize_before
        self.no_scale_embedding = no_scale_embedding
        self.layernorm_embedding = layernorm_embedding
        self.tie_word_embeddings = tie_word_embeddings
        self.deepnorm = deepnorm
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing
        self.moe_freq = moe_freq
        self.moe_top1_expert = moe_top1_expert
        self.moe_expert_count = moe_expert_count
        self.moe_gating_use_fp32 = moe_gating_use_fp32
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.use_xmoe = use_xmoe
        self.moe_second_expert_policy = moe_second_expert_policy
        self.moe_normalize_gate_prob_before_dropping = moe_normalize_gate_prob_before_dropping
        self.subln = subln
        self.xpos_rel_pos = xpos_rel_pos
        self.multiway = multiway
        self.fsdp = fsdp
        self.checkpoint_activations = checkpoint_activations
        self.no_output_layer = no_output_layer
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
