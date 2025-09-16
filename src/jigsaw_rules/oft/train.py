"""
Designed to train Qwen0.6b during submission
"""

from datasets import Dataset  # type: ignore
from trl import SFTTrainer  # type: ignore

from jigsaw_rules.oft.constants import (
    ChatTrainerConfig,
    InstructTrainerConfig,
)
from jigsaw_rules.utils import get_train_dataset


class QwenInstructTrainer:
    def __init__(
        self, data_path, model_path, save_path, lora_config, train_config
    ):
        self.data_path = data_path
        self.model_path = model_path
        self.save_path = save_path
        self.lora_config = lora_config
        self.train_config = train_config

    def run(self):
        dataframe = get_train_dataset(InstructTrainerConfig.model_type)

        train_dataset = Dataset.from_pandas(dataframe)

        lora_config = self.lora_config

        training_args = self.train_config

        trainer = SFTTrainer(
            self.model_path,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
        )

        trainer.train()
        trainer.save_model(self.save_path)


class QwenChatTrainer:
    def __init__(
        self, data_path, model_path, save_path, lora_config, train_config
    ):
        self.data_path = data_path
        self.model_path = model_path
        self.save_path = save_path
        self.lora_config = lora_config
        self.train_config = train_config

    def run(self):
        dataframe = get_train_dataset(ChatTrainerConfig.model_type)

        train_dataset = Dataset.from_pandas(dataframe)

        lora_config = self.lora_config

        training_args = self.train_config

        trainer = SFTTrainer(
            self.model_path,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
        )

        trainer.train()
        trainer.save_model(self.save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)

    args = parser.parse_args()

    if args.type == InstructTrainerConfig.model_type:
        trainer = QwenInstructTrainer(
            data_path=InstructTrainerConfig.data_path,
            model_path=InstructTrainerConfig.model_path,
            save_path=InstructTrainerConfig.lora_path,
            lora_config=InstructTrainerConfig.lora_config,
            train_config=InstructTrainerConfig.sft_config,
        )
        trainer.run()
    elif args.type == ChatTrainerConfig.model_type:
        chat_trainer = QwenChatTrainer(
            data_path=ChatTrainerConfig.data_path,
            model_path=ChatTrainerConfig.model_path,
            save_path=ChatTrainerConfig.lora_path,
            lora_config=ChatTrainerConfig.lora_config,
            train_config=ChatTrainerConfig.sft_config,
        )
        chat_trainer.run()
    else:
        raise AttributeError("Invalid train type")
