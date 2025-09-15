"""
Designed to do inference on Qwen0.5b (Instruction-Tuned) and Qwen14b (Chat Model) Int4
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
from vllm.lora.request import LoRARequest

from jigsaw_rules.constants import ChatConfig, InstructConfig
from jigsaw_rules.utils import build_dataset, build_dataset_chat


class RulesInference:
    def __init__(self, data_path, model_path, lora_path, save_path):
        self.data_path = data_path
        self.model_path = model_path
        self.lora_path = lora_path
        self.save_path = save_path

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


class ChatRulesInference(RulesInference):
    def get_dataset(self, df):
        df = build_dataset_chat(df)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)

    args = parser.parse_args()

    if args.type == InstructConfig.model_type:
        inference = RulesInference(
            data_path=InstructConfig.data_path,
            model_path=InstructConfig.model_path,
            lora_path=InstructConfig.lora_path,
            save_path=InstructConfig.out_file,
        )
    elif args.type == ChatConfig.model_type:
        inference = ChatRulesInference(
            data_path=ChatConfig.data_path,
            model_path=ChatConfig.model_path,
            lora_path=ChatConfig.lora_path,
            save_path=ChatConfig.out_file,
        )
    else:
        raise AttributeError("Invalid inference type")
    inference.run()
