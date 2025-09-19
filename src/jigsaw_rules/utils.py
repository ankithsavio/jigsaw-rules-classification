import random

import numpy as np
import pandas as pd  # type: ignore
from cleantext import clean  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from transformers import AutoTokenizer

from jigsaw_rules.constants import (
    ChatConfig,
    EmbeddingConfig,
    InstructConfig,
    RobertaConfig,
)

random.seed(42)
np.random.seed(42)


def build_prompt(row):
    return (
        f"{InstructConfig.base_prompt}\n"
        f"Subreddit: r/{row['subreddit']}\n"
        f"Rule: {row['rule']}\n"
        "Examples:\n"
        f"1) {row['positive_example']}\n"
        f"{InstructConfig.complete_phrase} Yes\n"
        f"2) {row['negative_example']}\n"
        f"{InstructConfig.complete_phrase} No\n"
        "---\n"
        f"Comment: {row['body']}\n"
        f"{InstructConfig.complete_phrase}"
    )


def build_prompt_chat(row, tokenizer):
    text = (
        f"r/{row['subreddit']}\n"
        f"Rule: {row['rule']}\n"
        "Examples:\n"
        f"1) {row['positive_example']}\n"
        f"{ChatConfig.complete_phrase} Yes\n"
        f"2) {row['negative_example']}\n"
        f"{ChatConfig.complete_phrase} No\n"
        "---\n"
        f"Comment: {row['body']}\n"
        f"{ChatConfig.complete_phrase}"
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
        + ChatConfig.complete_phrase
    )
    return prompt


def build_prompt_emb(row):
    return f"""r/{row["subreddit"]}\nComment: {row["body"]}"""


def cleaner(text):
    return clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        lang="en",
    )


def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = (
        pd.read_csv(f"{data_path}/test.csv")
        .sample(frac=1, random_state=42)  # shuffle
        .reset_index(drop=True)
    )

    flatten = []

    # ---------- process train data ----------
    train_df = train_dataset[
        [
            "body",
            "rule",
            "subreddit",
            "rule_violation",
            "positive_example_1",
            "positive_example_2",
            "negative_example_1",
            "negative_example_2",
        ]
    ].copy()

    # Randomly select positive and negative examples
    ## Undersampled
    train_df["positive_example"] = np.where(
        np.random.rand(len(train_df)) < 0.5,
        train_df["positive_example_1"],
        train_df["positive_example_2"],
    )
    train_df["negative_example"] = np.where(
        np.random.rand(len(train_df)) < 0.5,
        train_df["negative_example_1"],
        train_df["negative_example_2"],
    )

    # Delete original columns
    train_df.drop(
        columns=[
            "positive_example_1",
            "positive_example_2",
            "negative_example_1",
            "negative_example_2",
        ],
        inplace=True,
    )

    flatten.append(train_df)

    # ---------- process test data ----------

    ## test data is not labelled therefore use the example to create additional data for training
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[
                [
                    "rule",
                    "subreddit",
                    "positive_example_1",
                    "positive_example_2",
                    "negative_example_1",
                    "negative_example_2",
                ]
            ].copy()
            if violation_type == "positive":
                # body uses the current positive_example
                body_col = f"positive_example_{i}"
                other_positive_col = (
                    f"positive_example_{3 - i}"  # another positive
                )
                sub_dataset["body"] = sub_dataset[body_col]
                sub_dataset["positive_example"] = sub_dataset[
                    other_positive_col
                ]
                # negative_example randomly selected
                sub_dataset["negative_example"] = np.where(
                    np.random.rand(len(sub_dataset)) < 0.5,
                    sub_dataset["negative_example_1"],
                    sub_dataset["negative_example_2"],
                )
                sub_dataset["rule_violation"] = 1

            else:  # violation_type == "negative"
                body_col = f"negative_example_{i}"
                other_negative_col = f"negative_example_{3 - i}"
                sub_dataset["body"] = sub_dataset[body_col]
                sub_dataset["negative_example"] = sub_dataset[
                    other_negative_col
                ]
                sub_dataset["positive_example"] = np.where(
                    np.random.rand(len(sub_dataset)) < 0.5,
                    sub_dataset["positive_example_1"],
                    sub_dataset["positive_example_2"],
                )
                sub_dataset["rule_violation"] = 0

            # Delete the original candidate column
            sub_dataset.drop(
                columns=[
                    "positive_example_1",
                    "positive_example_2",
                    "negative_example_1",
                    "negative_example_2",
                ],
                inplace=True,
            )

            flatten.append(sub_dataset)

    # merge all DataFrame
    dataframe = pd.concat(flatten, axis=0)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    return dataframe


def build_dataset(dataframe):
    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    if "rule_violation" in dataframe:
        dataframe["completion"] = dataframe["rule_violation"].map(
            {
                1: InstructConfig.positive_answer,
                0: InstructConfig.negative_answer,
            }
        )

    return dataframe


def build_dataset_chat(dataframe):
    tokenizer = AutoTokenizer.from_pretrained(ChatConfig.model_path)
    dataframe["prompt"] = dataframe.apply(
        lambda row: build_prompt_chat(row, tokenizer), axis=1
    )

    if "rule_violation" in dataframe:
        dataframe["completion"] = dataframe["rule_violation"].map(
            {
                1: ChatConfig.positive_answer,
                0: ChatConfig.negative_answer,
            }
        )

    return dataframe


def build_dataset_emb(dataframe):
    # Semantic search
    dataframe["prompt"] = dataframe.apply(build_prompt_emb, axis=1)

    if EmbeddingConfig.clean_text:
        tqdm.pandas(desc="cleaner")
        dataframe["prompt"] = dataframe["prompt"].progress_apply(cleaner)

    if "rule_violation" in dataframe.columns:
        dataframe["rule_violation"] = dataframe["rule_violation"].map(
            {
                1: EmbeddingConfig.positive_answer,
                0: EmbeddingConfig.negative_answer,
            }
        )

    return dataframe


def build_dataset_emb_swift(dataframe):
    # Fine Tuning
    dataframe["messages"] = dataframe.apply(
        lambda row: [
            {"role": "system", "content": EmbeddingConfig.base_query},
            {"role": "user", "content": row["body"]},
        ],
        axis=1,
    )
    dataframe["positive_messages"] = dataframe.apply(
        lambda row: [
            [
                {
                    "role": "user",  # query should be closer to the positive example if it violates the rule
                    "content": row["positive_example"]
                    if row["rule_violation"] == 1
                    else row["negative_example"],
                }
            ]
        ],
        axis=1,
    )
    dataframe["negative_messages"] = dataframe.apply(
        lambda row: [
            [
                {
                    "role": "user",
                    "content": row["negative_example"]
                    if row["rule_violation"] == 1
                    else row["positive_example"],
                }
            ]
        ],
        axis=1,
    )
    return dataframe


def get_dataset_roberta(data_path):
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")

    test_df["positive"] = (
        test_df["positive_example_1"] + test_df["positive_example_2"]
    )
    test_df["negative"] = (
        test_df["negative_example_1"] + test_df["negative_example_2"]
    )

    test_df["val_pos_ex"] = 1
    test_df["val_neg_ex"] = 0

    df_add_pos = pd.DataFrame(
        test_df[["positive", "rule", "subreddit", "val_pos_ex"]]
    )
    df_add_pos.columns = ["body", "rule", "subreddit", "rule_violation"]
    df_add_neg = pd.DataFrame(
        test_df[["negative", "rule", "subreddit", "val_neg_ex"]]
    )
    df_add_neg.columns = ["body", "rule", "subreddit", "rule_violation"]

    df_add = (
        pd.concat([df_add_pos, df_add_neg], axis=0)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    train_df = pd.DataFrame(
        train_df[["body", "rule", "subreddit", "rule_violation"]].copy()
    )

    train_df = pd.concat([train_df, df_add], axis=0)

    train_df["input"] = (
        "rule:"
        + train_df["rule"]
        + "subreddit:"
        + train_df["subreddit"]
        + "body:"
        + train_df["body"]
    )

    test_df["input"] = (
        "rule:"
        + test_df["rule"]
        + "subreddit:"
        + test_df["subreddit"]
        + "body:"
        + test_df["body"]
    )

    return train_df, test_df


def get_train_dataset(model_type: str):  # train data optional during inference
    if model_type == InstructConfig.model_type:
        dataframe = get_dataframe_to_train(InstructConfig.data_path)
        dataset = build_dataset(dataframe)
    elif model_type == ChatConfig.model_type:
        dataframe = get_dataframe_to_train(ChatConfig.data_path)
        dataset = build_dataset_chat(dataframe)
    elif model_type == EmbeddingConfig.model_type:
        dataframe = get_dataframe_to_train(EmbeddingConfig.data_path)
        dataset = build_dataset_emb(dataframe)
        dataset = build_dataset_emb_swift(dataset)
    elif model_type == RobertaConfig.model_type:
        dataset, _ = get_dataset_roberta(RobertaConfig.data_path)
    else:
        raise AttributeError("Unknow model type")
    return dataset
