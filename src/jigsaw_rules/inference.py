"""
Designed to do inference with AutoRegressive and Bidirectional
"""

import multiprocessing as mp
import os
import random

import pandas as pd  # type: ignore
import torch
import vllm
from datasets import Dataset  # type: ignore
from logits_processor_zoo.vllm import (  # type: ignore
    MultipleChoiceLogitsProcessor,
)
from scipy.special import softmax  # type: ignore
from vllm.lora.request import LoRARequest

from jigsaw_rules.configs import (
    ChatConfig,
    DebertaConfig,
    InstructConfig,
    RobertaConfig,
)
from jigsaw_rules.dataset import RedditDataset
from jigsaw_rules.utils import (
    build_dataframe_chat,
    build_dataframe_deberta,
    build_dataframe_instruct,
    build_dataframe_roberta,
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
    """
    Manual DDP as accelerate doesnt work well with vLLM
    """

    def run_subset_device(self, test_dataset):
        """
        Run inference with single GPU
        """
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
        predictions["row_id"] = test_dataset["row_id"]
        return predictions

    def worker(self, device_id, test_dataset, return_dict):
        """
        subprocess worker function
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        print(
            f"[Worker {device_id}] Running on GPU {device_id}, data size={len(test_dataset)}"
        )

        preds = self.run_subset_device(test_dataset)
        return_dict[device_id] = preds

    def get_dataset(self):
        """
        get test data
        """
        if InstructConfig.test_file is None:
            dataframe = pd.read_csv(f"{self.data_path}/test.csv")
        else:
            dataframe = pd.read_csv(InstructConfig.test_file)

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

    def inference_with_data(self, data, return_preds=False):
        """
        Run inference on given data on 2x GPUs
        """
        # slice data
        mid = len(data) // 2
        df0 = data.iloc[:mid].reset_index(drop=True)
        df1 = data.iloc[mid:].reset_index(drop=True)
        df0 = Dataset.from_pandas(build_dataframe_instruct(df0))
        df1 = Dataset.from_pandas(build_dataframe_instruct(df1))
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
        if return_preds:
            return predictions[
                [
                    InstructConfig.negative_answer,
                    InstructConfig.positive_answer,
                ]
            ].to_numpy()

    def run(self):
        """
        Run inference on test data
        """
        test_dataframe = self.get_dataset()
        self.inference_with_data(test_dataframe)


class ChatEngine(JigsawInference):
    def get_dataset(self):
        """
        get test data
        """
        if ChatConfig.test_file is None:
            test_dataframe = pd.read_csv(f"{self.data_path}/test.csv")
        else:
            test_dataframe = pd.read_csv(ChatConfig.test_file)
        df = build_dataframe_chat(test_dataframe)
        return df

    def inference_with_data(self, data, return_preds=False):
        """
        Run inference on given data using Model Parallelism
        """

        dataset = Dataset.from_pandas(data)

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

        logprobs = [
            {
                lp.decoded_token: lp.logprob
                for lp in out.outputs[0].logprobs[0].values()  # type: ignore
            }
            for out in outputs
        ]
        logit_matrix = pd.DataFrame(logprobs)[
            [ChatConfig.positive_answer, ChatConfig.negative_answer]
        ]
        data = pd.concat([data, logit_matrix], axis=1)

        data[[ChatConfig.positive_answer, ChatConfig.negative_answer]] = data[
            [ChatConfig.positive_answer, ChatConfig.negative_answer]
        ].apply(lambda x: softmax(x.values), axis=1, result_type="expand")
        data["pred"] = data[ChatConfig.positive_answer]
        data["rule_violation"] = data["pred"]
        data[["row_id", "rule_violation"]].to_csv(self.save_path, index=False)
        print(f"Saved to {self.save_path}")

        if return_preds:
            return data[
                [ChatConfig.negative_answer, ChatConfig.positive_answer]
            ].to_numpy()

    def run(self):
        """
        Run inference on test data
        """
        dataset = self.get_dataset()

        self.inference_with_data(dataset)


class RobertaEngine(JigsawInference):
    def get_dataset(self):
        """
        get test data
        """
        _, df_test = build_dataframe_roberta()
        return df_test

    def inference_with_data(self, data, return_preds=False):
        """
        Run inference on given data using Data Parallelism
        """
        from transformers import (  # hackyfix : avoid cuda init in the parent process
            RobertaForSequenceClassification,
            RobertaTokenizer,
            Trainer,
            TrainingArguments,
        )

        tokenizer = RobertaTokenizer.from_pretrained(self.model_path)

        model = RobertaForSequenceClassification.from_pretrained(
            self.model_path
        ).to("cuda")  # type: ignore

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            report_to=[],
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
        )

        test_encodings = tokenizer(
            data["input"].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
        )

        test_dataset = RedditDataset(test_encodings)

        test_outputs = trainer.predict(test_dataset)
        full_probs = torch.nn.functional.softmax(
            torch.tensor(test_outputs.predictions), dim=1
        )

        submission_df = pd.DataFrame(
            {
                "row_id": data["row_id"],
                "rule_violation": full_probs[:, 1].numpy(),
            }
        )

        submission_df.to_csv(self.save_path, index=False)
        if return_preds:
            return full_probs.numpy()

    def run(self):
        """
        Run inference on test data
        """
        df_test = self.get_dataset()
        self.inference_with_data(df_test)


class DebertaEngine(JigsawInference):
    def get_dataset(self):
        """
        get test data
        """
        if DebertaConfig.test_file is None:
            dataframe = pd.read_csv(f"{self.data_path}/test.csv")
        else:
            dataframe = pd.read_csv(DebertaConfig.test_file)
        dataframe = build_dataframe_deberta(dataframe)
        return dataframe

    def inference_with_data(self, data, return_preds=False):
        """
        Run inference on given data using Data Parallelism
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import (  # hackyfix : avoid cuda init in the parent process
            DataCollatorWithPadding,
            DebertaV2ForSequenceClassification,
            DebertaV2Tokenizer,
            Trainer,
            TrainingArguments,
        )

        tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)
        collator = DataCollatorWithPadding(tokenizer)

        model = DebertaV2ForSequenceClassification.from_pretrained(
            self.model_path, num_labels=2
        )

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            report_to="none",
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collator,
        )

        test_encodings = tokenizer(
            data["input_text"].tolist(),
            truncation=True,
            max_length=256,
        )

        test_dataset = RedditDataset(test_encodings)

        predictions = trainer.predict(test_dataset)
        full_probs = torch.nn.functional.softmax(
            torch.tensor(predictions.predictions), dim=1
        )

        probs = full_probs[:, 1].numpy()

        submission_df = pd.DataFrame(
            {
                "row_id": data["row_id"],
                "rule_violation": probs,
            }
        )
        submission_df.to_csv(self.save_path, index=False)
        if return_preds:
            return full_probs.numpy()

    def run(self):
        """
        Run inference on test data
        """
        test_dataframe = self.get_dataset()
        self.inference_with_data(test_dataframe)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)

    args = parser.parse_args()

    os.environ["VLLM_USE_V1"] = "0"

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
            model_path=RobertaConfig.ckpt_path,
            save_path=RobertaConfig.out_file,
        )
        inference.run()
    elif args.type == DebertaConfig.model_type:
        inference = DebertaEngine(
            data_path=DebertaConfig.data_path,
            model_path=DebertaConfig.ckpt_path,
            save_path=DebertaConfig.out_file,
        )
        inference.run()
    else:
        raise AttributeError("Invalid inference type")
