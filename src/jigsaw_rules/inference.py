"""
Designed to do inference on test data
"""

import os

os.environ["VLLM_USE_V1"] = "0"

import multiprocessing as mp
import random

import pandas as pd  # type: ignore
import torch
import vllm
from datasets import Dataset  # type: ignore
from logits_processor_zoo.vllm import (  # type: ignore
    MultipleChoiceLogitsProcessor,
)
from sklearn.model_selection import train_test_split  # type: ignore
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)
from vllm.lora.request import LoRARequest

from jigsaw_rules.constants import ChatConfig, InstructConfig, RobertaConfig
from jigsaw_rules.dataset import RedditDataset
from jigsaw_rules.utils import (
    build_dataset,
    build_dataset_chat,
    get_dataset_roberta,
)


class JigsawInference:
    def __init__(self, data_path, model_path, lora_path=None, save_path=None):
        self.data_path = data_path
        self.model_path = model_path
        self.lora_path = lora_path if lora_path else ""
        self.save_path = save_path if save_path else ""

    def run(self):
        raise NotImplementedError


class InstructEngine(JigsawInference):
    def run_subset_device(self, df_slice):
        llm = vllm.LLM(
            self.model_path,
            quantization="gptq",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.98,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True,
            max_model_len=2836,
            disable_log_stats=True,
            enable_prefix_caching=True,
            enable_lora=True,
            max_lora_rank=64,
        )

        tokenizer = llm.get_tokenizer()
        mclp = MultipleChoiceLogitsProcessor(
            tokenizer,  # type: ignore
            choices=[
                InstructConfig.positive_answer,
                InstructConfig.negative_answer,
            ],
        )
        test_dataset = Dataset.from_pandas(build_dataset(df_slice))
        texts = test_dataset["prompt"]

        outputs = llm.generate(
            texts,
            vllm.SamplingParams(
                skip_special_tokens=True,
                max_tokens=1,
                logits_processors=[mclp],
                logprobs=2,
            ),
            use_tqdm=True,
            lora_request=LoRARequest("default", 1, self.lora_path),
        )

        log_probs = [
            {
                lp.decoded_token: lp.logprob
                for lp in out.outputs[0].logprobs[0].values()  # type: ignore
            }
            for out in outputs
        ]
        predictions = pd.DataFrame(log_probs)[
            [InstructConfig.positive_answer, InstructConfig.negative_answer]
        ]
        predictions["row_id"] = df_slice["row_id"].values
        return predictions

    def worker(self, device_id, test_dataset, return_dict):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        print(
            f"[Worker {device_id}] Running on GPU {device_id}, data size={len(test_dataset)}"
        )

        preds = self.run_subset_device(test_dataset)
        return_dict[device_id] = preds

    def get_dataset(self):
        dataframe = pd.read_csv(f"{self.data_path}/test.csv")

        # randomly selected examples
        dataframe["positive_example"] = dataframe.apply(
            lambda row: random.choice(
                [row["positive_example_1"], row["positive_example_2"]]
            ),
            axis=1,
        )
        dataframe["negative_example"] = dataframe.apply(
            lambda row: random.choice(
                [row["negative_example_1"], row["negative_example_2"]]
            ),
            axis=1,
        )
        dataframe = dataframe.drop(
            columns=[
                "positive_example_1",
                "positive_example_2",
                "negative_example_1",
                "negative_example_2",
            ],
            errors="ignore",
        )
        return dataframe

    def run(self):
        """
        Data Parallelism
        """
        test_dataframe = self.get_dataset()
        # slice data
        mid = len(test_dataframe) // 2
        df0 = test_dataframe.iloc[:mid].reset_index(drop=True)
        df1 = test_dataframe.iloc[mid:].reset_index(drop=True)
        manager = mp.Manager()
        return_dict = manager.dict()

        # Two processes in parallel
        p0 = mp.Process(target=self.worker, args=(0, df0, return_dict))
        p1 = mp.Process(target=self.worker, args=(1, df1, return_dict))
        p0.start()
        p1.start()
        p0.join()
        p1.join()

        # merge results
        predictions = pd.concat(
            [return_dict[0], return_dict[1]], ignore_index=True
        )

        # build submission
        submission = predictions[
            ["row_id", InstructConfig.positive_answer]
        ].rename(columns={InstructConfig.positive_answer: "rule_violation"})

        submission.to_csv(self.save_path, index=False)
        print(f"Saved to {self.save_path}")


class ChatEngine(JigsawInference):
    def get_dataset(self, dataframe):
        # randomly selected examples
        dataframe["positive_example"] = dataframe.apply(
            lambda row: random.choice(
                [row["positive_example_1"], row["positive_example_2"]]
            ),
            axis=1,
        )
        dataframe["negative_example"] = dataframe.apply(
            lambda row: random.choice(
                [row["negative_example_1"], row["negative_example_2"]]
            ),
            axis=1,
        )
        dataframe = dataframe.drop(
            columns=[
                "positive_example_1",
                "positive_example_2",
                "negative_example_1",
                "negative_example_2",
            ],
            errors="ignore",
        )

        df = build_dataset_chat(dataframe)
        return df

    def run(self):
        """
        Model Parallelism
        """
        test_dataframe = pd.read_csv(f"{self.data_path}/test.csv")
        dataset = self.get_dataset(test_dataframe)
        dataset = Dataset.from_pandas(dataset)

        llm = vllm.LLM(
            self.model_path,
            quantization="gptq",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True,
            max_model_len=2836,
            disable_log_stats=True,
            enable_prefix_caching=True,
            enable_lora=True,
            max_lora_rank=32,
        )

        tokenizer = llm.get_tokenizer()
        mclp = MultipleChoiceLogitsProcessor(
            tokenizer,  # type: ignore
            choices=[
                ChatConfig.positive_answer,
                ChatConfig.negative_answer,
            ],
        )
        texts = dataset["prompt"]

        outputs = llm.generate(
            texts,
            vllm.SamplingParams(
                skip_special_tokens=True,
                max_tokens=1,
                logits_processors=[mclp],
                logprobs=2,
            ),
            use_tqdm=True,
            lora_request=LoRARequest("default", 1, self.lora_path),
        )

        log_probs = [
            {
                lp.decoded_token: lp.logprob
                for lp in out.outputs[0].logprobs[0].values()  # type: ignore
            }
            for out in outputs
        ]
        predictions = pd.DataFrame(log_probs)[
            [ChatConfig.positive_answer, ChatConfig.negative_answer]
        ]
        predictions["row_id"] = test_dataframe[
            "row_id"
        ]  # dataset so no need to use .values

        # build submission
        submission = predictions[
            [
                "row_id",
                ChatConfig.positive_answer,
            ]  # some people normalize logits against both answer yes or no and then use yes for submission
        ].rename(columns={ChatConfig.positive_answer: "rule_violation"})

        submission.to_csv(self.save_path, index=False)
        print(f"Saved to {self.save_path}")


class RobertaEngine(JigsawInference):
    def get_dataset(self):
        df_train, df_test = get_dataset_roberta(RobertaConfig.data_path)
        return df_train, df_test

    def run(self):
        df_train, df_test = self.get_dataset()
        X = df_train["input"].tolist()
        y = df_train["rule_violation"].tolist()

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
            logging_strategy="steps",
            logging_steps=10,
            logging_dir="./logs",
            report_to=[],
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        test_encodings = tokenizer(
            df_test["input"].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
        )

        dummy_labels = [0] * len(df_test)
        test_dataset = RedditDataset(test_encodings, dummy_labels)

        test_outputs = trainer.predict(test_dataset)
        probs = torch.nn.functional.softmax(
            torch.tensor(test_outputs.predictions), dim=1
        )[:, 1].numpy()

        submission_df = pd.DataFrame(
            {"row_id": df_test["row_id"], "rule_violation": probs}
        )

        submission_df.to_csv(self.save_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)

    args = parser.parse_args()

    if args.type == InstructConfig.model_type:
        inference: JigsawInference = InstructEngine(
            data_path=InstructConfig.data_path,
            model_path=InstructConfig.model_path,
            lora_path=InstructConfig.lora_path,
            save_path=InstructConfig.out_file,
        )
        inference.run()
    elif args.type == ChatConfig.model_type:
        inference = ChatEngine(
            data_path=ChatConfig.data_path,
            model_path=ChatConfig.model_path,
            lora_path=ChatConfig.lora_path,
            save_path=ChatConfig.out_file,
        )
        inference.run()
    elif args.type == RobertaConfig.model_type:
        inference = RobertaEngine(
            data_path=RobertaConfig.data_path,
            model_path=RobertaConfig.model_path,
            save_path=RobertaConfig.out_file,
        )
        inference.run()
    else:
        raise AttributeError("Invalid inference type")
