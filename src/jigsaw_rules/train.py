"""
Designed to train Qwen0.6b during submission
"""

import pandas as pd  # type: ignore
from datasets import Dataset  # type: ignore
from peft import LoraConfig
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    SimilarityFunction,
    TripletEvaluator,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sklearn.model_selection import train_test_split  # type: ignore
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.utils.import_utils import is_torch_bf16_gpu_available
from trl import SFTConfig, SFTTrainer  # type: ignore

from jigsaw_rules.constants import InstructConfig, RobertaConfig, e5Config
from jigsaw_rules.dataset import RedditDataset
from jigsaw_rules.utils import get_train_dataset


class JigsawTrainer:
    def __init__(self, data_path, model_path, save_path):
        self.data_path = data_path
        self.model_path = model_path
        self.save_path = save_path

    def run(self):
        raise NotImplementedError


class Instruct(JigsawTrainer):
    def run(self):
        dataframe = get_train_dataset(InstructConfig.model_type)

        if InstructConfig.use_subset:
            dataframe = dataframe.sample(
                frac=InstructConfig.subset, random_state=42
            ).reset_index(drop=True)

        train_dataset = Dataset.from_pandas(dataframe)

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


class RobertaBase(JigsawTrainer):
    def run(self):
        dataframe = get_train_dataset(RobertaConfig.model_type)

        if RobertaConfig.use_subset:
            dataframe = dataframe.sample(
                frac=RobertaConfig.subset, random_state=42
            ).reset_index(drop=True)

        X = dataframe["input"].tolist()
        y = dataframe["rule_violation"].tolist()

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
            num_train_epochs=6,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            disable_tqdm=False,
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


class e5Base(JigsawTrainer):
    def run(self):
        dataframe = get_train_dataset(e5Config.model_type)
        dataframe = pd.DataFrame(
            dataframe[
                ["anchor", "positive_example", "negative_example"]
            ].copy()
        )
        dataframe.columns = ["anchor", "positive", "negative"]

        if e5Config.use_subset:
            dataframe = dataframe.sample(
                frac=e5Config.subset, random_state=42
            ).reset_index(drop=True)

        dataset = Dataset.from_pandas(dataframe).train_test_split(
            test_size=0.01, seed=42
        )

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        model = SentenceTransformer(self.model_path)
        loss = MultipleNegativesRankingLoss(model)

        args = SentenceTransformerTrainingArguments(
            output_dir=self.save_path,
            num_train_epochs=16,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            fp16=True,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="no",
            report_to="none",
        )

        evaluator = TripletEvaluator(
            anchors=list(test_dataset["anchor"]),
            positives=list(test_dataset["positive"]),
            negatives=list(test_dataset["negative"]),
            main_distance_function=SimilarityFunction.COSINE,
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,  # only 3 columns, anchor, positive, negative
            eval_dataset=test_dataset,
            loss=loss,
            evaluator=evaluator,
        )

        trainer.train()
        evaluator(model)
        trainer.save_model(self.save_path)


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
    elif args.type == e5Config.model_type:
        trainer = e5Base(
            data_path=e5Config.data_path,
            model_path=e5Config.model_path,
            save_path=e5Config.ckpt_path,
        )
        trainer.run()
