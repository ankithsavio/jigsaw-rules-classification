from peft import LoraConfig
from transformers.utils.import_utils import is_torch_bf16_gpu_available
from trl import SFTConfig  # type: ignore


class InstructTrainerConfig:
    model_type = "instruct"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    model_path = "/kaggle/input/qwen2.5/transformers/0.5b-instruct-gptq-int4/1"
    lora_path = "output/"
    positive_answer = "Yes"
    negative_answer = "No"
    complete_phrase = "Answer:"
    base_prompt = """You are given a comment from reddit and a rule. Your task is to classify whether the comment violates the rule. Only respond Yes/No."""
    out_file = "submission_instruct.csv"
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=1e-4,  # keep high, lora usually likes high.
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=is_torch_bf16_gpu_available(),
        fp16=not is_torch_bf16_gpu_available(),
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to="none",
        completion_only_loss=True,
        packing=False,
        remove_unused_columns=False,
    )


class ChatTrainerConfig:
    model_type = "chat"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    model_path = "/kaggle/input/qwen2.5/transformers/14b-instruct-gptq-int4/1"
    lora_path = "/kaggle/input/lora_14b_gptq_1epoch_r32/keras/default/1"
    positive_answer = "Yes"
    negative_answer = "No"
    complete_phrase = "Answer:"
    base_prompt = """You are given a comment on reddit. Your task is to classify if it violates the given rule. Only respond Yes/No."""
    out_file = "submission_chat.csv"
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=1e-4,  # keep high, lora usually likes high.
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=is_torch_bf16_gpu_available(),
        fp16=not is_torch_bf16_gpu_available(),
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to="none",
        completion_only_loss=True,
        packing=False,
        remove_unused_columns=False,
    )


class EmbedTrainerConfig:
    model_type = "embed"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    model_path = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
    lora_path = "/kaggle/input/qwen3-8b-embedding"  # only named as 8b but actually uses 0.6b
    positive_answer = 1
    negative_answer = -1
    base_query = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    clean_text = True
    top_k = 2000
    batch_size = 128
    out_file = "submission_embedding.csv"
