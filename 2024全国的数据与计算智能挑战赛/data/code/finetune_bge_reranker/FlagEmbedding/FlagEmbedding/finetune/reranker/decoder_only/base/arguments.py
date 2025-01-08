from typing import List
from dataclasses import dataclass, field

from FlagEmbedding.abc.finetune.reranker import AbsRerankerModelArguments


def default_target_modules() -> List[int]:
    return ['v_proj', 'q_proj', 'k_proj', 'gate_proj', 'down_proj', 'o_proj', 'up_proj']


@dataclass
class RerankerModelArguments(AbsRerankerModelArguments):
    use_lora: bool = field(
        default=True,
        metadata={"help": "If passed, will use LORA (low-rank parameter-efficient training) to train the model."}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."}
    )
    target_modules: List[str] = field(
        default_factory=default_target_modules,
        metadata={"help": "The target modules to apply LORA."}
    )
    modules_to_save: List[str] = field(
        default=None,
        metadata={"help": "List of modules that should be saved in the final checkpoint."}
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "If passed, will use flash attention to train the model."}
    )
    # use_slow_tokenizer: bool = field(
    #     default=False,
    #     metadata={"help": "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."}
    # )
    from_peft: str = field(
        default=None
    )
    raw_peft: List[str] = field(
        default=None
    )
    
    save_merged_lora_model: bool = field(
        default=False,
        metadata={"help": "If passed, will merge the lora modules and save the entire model."}
    )
