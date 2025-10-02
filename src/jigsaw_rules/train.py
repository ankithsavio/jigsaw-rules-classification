"""
Designed for Test Time Fine Tuning (TTFT)
"""

import os

import pandas as pd  # type: ignore
from datasets import Dataset  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from jigsaw_rules.configs import (
    DebertaConfig,
    E5Config,
    InstructConfig,
    RobertaConfig,
)
from jigsaw_rules.dataset import RedditDataset
from jigsaw_rules.utils import get_train_dataframe


class JigsawTrainer:
    def __init__(self, data_path, model_path, save_path):
        self.data_path = data_path
        self.model_path = model_path
        self.save_path = save_path

    def run(self):
        raise NotImplementedError


class Instruct(JigsawTrainer):
    def train_with_data(self, data):
        """
        Run Trainer on data
        """
        from peft import LoraConfig
        from transformers.utils.import_utils import is_torch_bf16_gpu_available
        from trl import SFTConfig, SFTTrainer  # type: ignore

        train_dataset = Dataset.from_pandas(data)

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

    def run(self):
        """
        Run Trainer on data determined by Config.data_path
        """
        dataframe = get_train_dataframe(InstructConfig.model_type)

        self.train_with_data(dataframe)


class RobertaBase(JigsawTrainer):
    def train_with_data(self, data):
        """
        Run Trainer on data
        """
        from transformers import (
            RobertaForSequenceClassification,
            RobertaTokenizer,
            Trainer,
            TrainingArguments,
        )

        X = data["input"].tolist()
        y = data["rule_violation"].tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        tokenizer = RobertaTokenizer.from_pretrained(RobertaConfig.model_path)
        model = RobertaForSequenceClassification.from_pretrained(
            RobertaConfig.model_path
        )

        train_encodings = tokenizer(
            X_train, truncation=True, padding=True, max_length=512
        )
        val_encodings = tokenizer(
            X_val, truncation=True, padding=True, max_length=512
        )

        train_dataset = RedditDataset(train_encodings, y_train)
        val_dataset = RedditDataset(val_encodings, y_val)

        model = RobertaForSequenceClassification.from_pretrained(
            RobertaConfig.model_path
        ).to("cuda")  # type: ignore

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=2,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            warmup_ratio=0.1,
            weight_decay=0.01,
            report_to="none",
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
        trainer.save_model(self.save_path)
        tokenizer.save_pretrained(self.save_path)

    def run(self):
        """
        Run Trainer on data determined by Config.data_path
        """
        dataframe, _ = get_train_dataframe(RobertaConfig.model_type)
        self.train_with_data(dataframe)


class E5Base(JigsawTrainer):
    def train_with_data(self, data):
        """
        Run Trainer on data
        """
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import MultipleNegativesRankingLoss

        data = pd.DataFrame(
            data[["anchor", "positive_example", "negative_example"]].copy()
        )
        data.columns = ["anchor", "positive", "negative"]

        dataset = Dataset.from_pandas(data).train_test_split(
            test_size=0.01, seed=42
        )

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        model = SentenceTransformer(self.model_path)
        loss = MultipleNegativesRankingLoss(model)

        args = SentenceTransformerTrainingArguments(
            output_dir="./results",
            num_train_epochs=2,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            warmup_ratio=0.1,
            weight_decay=0.01,
            report_to="none",
            save_strategy="no",
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,  # only 3 columns, anchor, positive, negative
            eval_dataset=test_dataset,
            loss=loss,
        )

        trainer.train()
        trainer.save_model(self.save_path)

    def run(self):
        """
        Run Trainer on data determined by Config.data_path
        """
        dataframe = get_train_dataframe(E5Config.model_type)
        self.train_with_data(dataframe)


class DebertaBase(JigsawTrainer):
    def train_with_data(self, data):
        """
        Run Trainer on data
        """
        from transformers import (
            DataCollatorWithPadding,
            DebertaV2ForSequenceClassification,
            DebertaV2Tokenizer,
            Trainer,
            TrainingArguments,
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)
        collator = DataCollatorWithPadding(tokenizer)

        data.drop_duplicates(
            subset=["body", "rule"], keep="first", inplace=True
        )

        train_encodings = tokenizer(
            data["input_text"].tolist(),
            truncation=True,
            max_length=512,
        )

        train_labels = data["rule_violation"].tolist()
        train_dataset = RedditDataset(train_encodings, train_labels)

        model = DebertaV2ForSequenceClassification.from_pretrained(
            self.model_path, num_labels=2
        )

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            warmup_ratio=0.1,
            weight_decay=0.01,
            report_to="none",
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )
        trainer.train()
        trainer.save_model(self.save_path)
        tokenizer.save_pretrained(self.save_path)

    def run(self):
        """
        Run Trainer on data determined by Config.data_path
        """
        dataframe = get_train_dataframe(DebertaConfig.model_type)
        self.train_with_data(dataframe)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)

    args = parser.parse_args()

    if args.type == InstructConfig.model_type:
        trainer: JigsawTrainer = Instruct(
            data_path=InstructConfig.data_path,
            model_path=InstructConfig.model_path,
            save_path=InstructConfig.lora_path,
        )
        trainer.run()
    elif args.type == RobertaConfig.model_type:
        trainer = RobertaBase(
            data_path=RobertaConfig.data_path,
            model_path=RobertaConfig.model_path,
            save_path=RobertaConfig.ckpt_path,
        )
        trainer.run()
    elif args.type == E5Config.model_type:
        trainer = E5Base(
            data_path=E5Config.data_path,
            model_path=E5Config.model_path,
            save_path=E5Config.ckpt_path,
        )
        trainer.run()
    elif args.type == DebertaConfig.model_type:
        trainer = DebertaBase(
            data_path=DebertaConfig.data_path,
            model_path=DebertaConfig.model_path,
            save_path=DebertaConfig.ckpt_path,
        )
        trainer.run()
    else:
        raise AttributeError("Invalid inference type")
