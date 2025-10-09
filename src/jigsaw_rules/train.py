"""
Designed for Test Time Fine Tuning (TTFT)
"""

import os

import pandas as pd  # type: ignore
from datasets import Dataset  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from jigsaw_rules.configs import (
    BgeConfig,
    DebertaConfig,
    E5Config,
    ExtendedDebertaConfig,
    InstructConfig,
    ModernBERTConfig,
    RobertaConfig,
)
from jigsaw_rules.dataset import ExtendedRedditDataset, FusionCollator, RedditDataset
from jigsaw_rules.models import DebertaWithFusion
from jigsaw_rules.utils import get_train_dataframe

from jigsaw_rules.inference import ChatEngine, ChatConfig


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


class BgeBase(JigsawTrainer):
    def train_with_data(self, data):
        """
        Run Trainer on data
        """
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            models,
        )
        from sentence_transformers.losses import TripletLoss

        word_embedding_model = models.Transformer(
            self.model_path, max_seq_length=128, do_lower_case=True
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode="mean",
        )
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model]
        )
        loss = TripletLoss(model=model, triplet_margin=0.25)

        args = SentenceTransformerTrainingArguments(
            output_dir="./results",
            num_train_epochs=2,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            warmup_steps=0,
            report_to="none",
            save_strategy="no",
            fp16=True,
            max_grad_norm=1.0,
            dataloader_drop_last=False,
            gradient_accumulation_steps=1,
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=data,
            loss=loss,
        )

        trainer.train()
        trainer.save_model(self.save_path)

    def run(self):
        """
        Run Trainer on data determined by Config.data_path
        """
        dataframe = get_train_dataframe(BgeConfig.model_type)
        dataset = Dataset.from_pandas(dataframe)
        self.train_with_data(dataset)


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


class ExtendedDebertaBase(JigsawTrainer):
    def train_with_data(self, data):
        """
        Train a fusion head on top of DeBERTa logits + two extra probabilities.
        Expected columns in `data`:
          - 'input_text' (str)
          - 'rule_violation' (int 0/1)
          - 'p1' (float in [0,1])  # first external probability
          - 'p2' (float in [0,1])  # second external probability
        """

        from transformers import (
            DebertaV2Tokenizer,
            TrainingArguments,
            Trainer,
            DebertaV2Config,
        )
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # --- tokenizer & de-dup ---
        tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)
        
        ## TODO: Check this later I don't want to suffle by mistake
        data = data.drop_duplicates(subset=["body", "rule"], keep="first").reset_index(drop=True)

        # --- tokenize ---
        enc = tokenizer(
            data["input_text"].tolist(),
            truncation=True,
            max_length=512,
            padding=False,   # padding handled by collator
        )

        labels = data["rule_violation"].astype(int).tolist()
        extras = data[["p1", "p2"]].astype(float).values  # shape [N,2]

        dataset = ExtendedRedditDataset(enc, labels, extras)
        collator = FusionCollator(tokenizer)

        # --- fusion model (freeze base by default) ---
        cfg = DebertaV2Config.from_pretrained(self.model_path, num_labels=2)
        model = DebertaWithFusion(
            cfg,
            base_model_name=self.model_path,
            freeze_base=False,      # set False to fine-tune DeBERTa too
            extra_size=2,
            hidden_size=ExtendedDebertaConfig.hidden_size,
        )

        # --- training args ---
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            learning_rate=2e-5,  # slightly higher since only small head trains
            per_device_train_batch_size=8,
            warmup_ratio=0.1,
            weight_decay=0.01,
            report_to="none",
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
        )

        trainer.train()
        trainer.save_model(self.save_path)        # saves config + weights
        tokenizer.save_pretrained(self.save_path) # keeps tokenizer with the model
    
    def run(self):
        """
        Run Trainer on data determined by Config.data_path
        """
        dataframe = get_train_dataframe(ExtendedDebertaConfig.model_type)
        
        ## Adding extra probabilities for fusion training
        ## Brute force solution run this in the kaggle notebook first
        ## to generate the probabilities and save them to a csv
        ## then read them here and merge with the dataframe
        # because inserting coding to clean the gpu can cause troubles
        # inference = ChatEngine(
        #     data_path=ChatConfig.data_path,
        #     model_path=ChatConfig.model_path,
        #     lora_path=ChatConfig.lora_path,
        #     save_path=ChatConfig.out_file,
        # )
        # inference.run()

        # TODO: I need to loaded the data from training not inference
        chat_data = pd.read_csv(ChatConfig.out_file)
        chat_logits = chat_data[["p1", "p2"]]
        dataframe = pd.concat([dataframe, chat_logits], axis=1)

        self.train_with_data(dataframe)

class ModernBERTBase(JigsawTrainer):
    def train_with_data(self, data):
        """
        Run Trainer on data
        """
        from transformers import (
            AutoTokenizer,
            DataCollatorWithPadding,
            ModernBertForSequenceClassification,
            Trainer,
            TrainingArguments,
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        collator = DataCollatorWithPadding(tokenizer)

        data.drop_duplicates(
            subset=["body", "rule"], keep="first", inplace=True
        )

        train_encodings = tokenizer(
            data["input_text"].tolist(),
            truncation=True,
            max_length=8192,
        )

        train_labels = data["rule_violation"].tolist()
        train_dataset = RedditDataset(train_encodings, train_labels)

        model = ModernBertForSequenceClassification.from_pretrained(
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
    elif args.type == BgeConfig.model_type:
        trainer = BgeBase(
            data_path=BgeConfig.data_path,
            model_path=BgeConfig.model_path,
            save_path=BgeConfig.ckpt_path,
        )
        trainer.run()
    elif args.type == DebertaConfig.model_type:
        trainer = DebertaBase(
            data_path=DebertaConfig.data_path,
            model_path=DebertaConfig.model_path,
            save_path=DebertaConfig.ckpt_path,
        )
        trainer.run()
    elif args.type == ExtendedDebertaConfig.model_type:
        trainer = ExtendedDebertaBase(
            data_path=ExtendedDebertaConfig.data_path,
            model_path=ExtendedDebertaConfig.model_path,
            save_path=ExtendedDebertaConfig.ckpt_path,
        )
        trainer.run()
    elif args.type == ModernBERTConfig.model_type:
        trainer = ModernBERTBase(
            data_path=ModernBERTConfig.data_path,
            model_path=ModernBERTConfig.model_path,
            save_path=ModernBERTConfig.ckpt_path,
        )
        trainer.run()
    else:
        raise AttributeError("Invalid inference type")
