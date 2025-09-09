"""
Designed to train Qwen0.6b during submission
"""

from datasets import Dataset  # type: ignore
from peft import LoraConfig
from transformers.utils.import_utils import is_torch_bf16_gpu_available
from trl import SFTConfig, SFTTrainer  # type: ignore

from jigsaw_rules.constants import InstructConfig
from jigsaw_rules.utils import build_dataset, get_dataframe_to_train


class RulesTrainer:
    def __init__(self, data_path, model_path, save_path):
        self.data_path = data_path
        self.model_path = model_path
        self.save_path = save_path

    def run(self):
        dataframe = get_dataframe_to_train(self.data_path)

        train_dataset = Dataset.from_pandas(build_dataset(dataframe))

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

        training_args = SFTConfig(
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

        trainer = SFTTrainer(
            self.model_path,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
        )

        trainer.train()
        trainer.save_model(self.save_path)


if __name__ == "__main__":
    trainer = RulesTrainer(
        data_path=InstructConfig.data_path,
        model_path=InstructConfig.model_path,
        save_path=InstructConfig.lora_path,
    )

    trainer.run()
