"""
Designed to do inference on Qwen0.5b (Instruction-Tuned) and Qwen14b (Chat Model) Int4
"""

import os

os.environ["VLLM_USE_V1"] = "0"

import multiprocessing as mp
import random

import pandas as pd  # type: ignore
import vllm
from datasets import Dataset  # type: ignore
from logits_processor_zoo.vllm import (  # type: ignore
    MultipleChoiceLogitsProcessor,
)
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest

from jigsaw_rules.constants import ChatConfig, InstructConfig
from jigsaw_rules.utils import build_dataset


class RulesInference:
    def __init__(self, data_path, model_path, lora_path, save_path):
        self.data_path = data_path
        self.model_path = model_path
        self.lora_path = lora_path
        self.save_path = save_path

    def run_subset_device(self, test_dataset):
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
        predictions["row_id"] = test_dataset["row_id"].values
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
        return build_dataset(dataframe)

    def run(self):
        test_dataframe = self.get_dataset()
        # slice data
        mid = len(test_dataframe) // 2
        df0 = test_dataframe.iloc[:mid].reset_index(drop=True)
        df1 = test_dataframe.iloc[mid:].reset_index(drop=True)
        df0 = Dataset.from_pandas(df0)
        df1 = Dataset.from_pandas(df1)
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
    def get_dataset(self):
        df = pd.read_csv(f"{self.data_path}/test.csv")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        prompts = []
        for i, row in df.iterrows():
            text = (
                f"r/{row.subreddit}\n"
                f"Rule: {row.rule}\n"
                f"1) {row.positive_example_1}\n"
                "Violation: Yes\n"
                f"2) {row.positive_example_2}\n"
                "Violation: Yes\n"
                f"3) {row.negative_example_1}\n"
                "Violation: No\n"
                f"4) {row.negative_example_2}\n"
                "Violation: No\n"
                f"5) {row.body}\n"
            )

            messages = [
                {"role": "system", "content": ChatConfig.base_prompt},
                {"role": "user", "content": text},
            ]

            prompt = (
                tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                + "Answer:"
            )
            prompts.append(prompt)

        df["prompt"] = prompts
        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)

    args = parser.parse_args()

    if args.type == "instruction":
        inference = RulesInference(
            data_path=InstructConfig.data_path,
            model_path=InstructConfig.model_path,
            lora_path=InstructConfig.lora_path,
            save_path=InstructConfig.out_file,
        )
        inference.run()
    elif args.type == "chat":
        inference = ChatRulesInference(
            data_path=ChatConfig.data_path,
            model_path=ChatConfig.model_path,
            lora_path=ChatConfig.lora_path,
            save_path=ChatConfig.out_file,
        )
        inference.run()
