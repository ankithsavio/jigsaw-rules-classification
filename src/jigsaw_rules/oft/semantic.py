import os
import subprocess


def main():
    env = os.environ.copy()
    env["NPROC_PER_NODE"] = "2"

    # Command as list
    cmd = [
        "swift",
        "sft",
        "--model",
        "Qwen/Qwen3-Reranker-4B",
        "--task_type",
        "generative_reranker",
        "--loss_type",
        "generative_reranker",
        "--train_type",
        "full",
        "--dataset",
        "MTEB/scidocs-reranking",
        "--split_dataset_ratio",
        "0.05",
        "--eval_strategy",
        "steps",
        "--output_dir",
        "output",
        "--eval_steps",
        "100",
        "--num_train_epochs",
        "1",
        "--save_steps",
        "200",
        "--per_device_train_batch_size",
        "2",
        "--per_device_eval_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "8",
        "--learning_rate",
        "6e-6",
        "--label_names",
        "labels",
        "--dataloader_drop_last",
        "true",
    ]

    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
