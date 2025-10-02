import random
import re

import numpy as np
import pandas as pd  # type: ignore
from cleantext import clean  # type: ignore
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from jigsaw_rules.configs import (
    BgeConfig,
    ChatConfig,
    DebertaConfig,
    E5Config,
    EmbeddingConfig,
    InstructConfig,
    RobertaConfig,
)


class DataframeFactory:
    _builders = {}  # type: ignore

    @classmethod
    def register(cls, model_type):
        def wrapper(func):
            cls._builders[model_type] = func  # register function
            return func

        return wrapper

    @classmethod
    def build(cls, model_type, *args, **kwargs):
        if model_type not in cls._builders:
            raise AttributeError(f"Unknown model type {model_type}")
        return cls._builders[model_type](*args, **kwargs)


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


def url_to_semantics(text):
    if not isinstance(text, str):
        return ""

    urls = re.findall(r"https?://[^\s/$.?#].[^\s]*", text)
    if not urls:
        return ""

    all_semantics = []
    seen_semantics = set()

    for url in urls:
        url_lower = url.lower()

        domain_match = re.search(
            r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url_lower
        )
        if domain_match:
            full_domain = domain_match.group(1)
            parts = full_domain.split(".")
            for part in parts:
                if part and part not in seen_semantics and len(part) > 3:
                    all_semantics.append(f"domain:{part}")
                    seen_semantics.add(part)

        path = re.sub(
            r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower
        )
        path_parts = [
            p for p in re.split(r"[/_.-]+", path) if p and p.isalnum()
        ]

        for part in path_parts:
            part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if (
                part_clean
                and part_clean not in seen_semantics
                and len(part_clean) > 3
            ):
                all_semantics.append(f"path:{part_clean}")
                seen_semantics.add(part_clean)

    if not all_semantics:
        return ""
    return f"\nURL Keywords: {' '.join(all_semantics)}"


def get_dataframe_to_train(data_path, subset=None):
    """
    Process test data to create additional training samples
    """
    if subset is None:
        test_dataset = pd.read_csv(f"{data_path}/test.csv")
    else:
        test_dataset = (
            pd.read_csv(f"{data_path}/test.csv")
            .sample(frac=subset, random_state=42)
            .reset_index(drop=True)
        )

    flatten = []

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
    dataframe = dataframe.sample(frac=1, random_state=42).reset_index(
        drop=True  # shuffle
    )

    return dataframe


def get_dataframe_to_train_semantic(
    data_path, include_train=True, subset=None
):
    if subset is None:
        test_dataset = (
            pd.read_csv(f"{data_path}/test.csv")
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
    else:
        test_dataset = (
            pd.read_csv(f"{data_path}/test.csv")
            .sample(frac=0.6, random_state=42)
            .reset_index(drop=True)
        )

    flatten = []
    if include_train:
        train_dataset = pd.read_csv(f"{data_path}/train.csv")
        flatten.append(
            train_dataset[["body", "rule", "subreddit", "rule_violation"]]
        )

    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[
                [f"{violation_type}_example_{i}", "rule", "subreddit"]
            ].copy()
            sub_dataset = sub_dataset.rename(
                columns={f"{violation_type}_example_{i}": "body"}
            )
            sub_dataset["rule_violation"] = (
                1 if violation_type == "positive" else 0
            )
            flatten.append(sub_dataset)

    dataframe = pd.concat(flatten, axis=0)
    dataframe = dataframe.drop_duplicates(ignore_index=True)
    return dataframe


@DataframeFactory.register(InstructConfig.model_type)
def build_dataframe_instruct(dataframe=None, is_train=False):
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

    if dataframe is None:  # training
        if not InstructConfig.use_subset:
            dataframe = get_dataframe_to_train(InstructConfig.data_path)
        else:
            dataframe = get_dataframe_to_train(
                InstructConfig.data_path,
                InstructConfig.subset,
            )

    if is_train and InstructConfig.include_train:
        train_df = pd.read_csv(f"{InstructConfig.data_path}/train.csv")
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
        train_df.drop(
            columns=[
                "positive_example_1",
                "positive_example_2",
                "negative_example_1",
                "negative_example_2",
            ],
            inplace=True,
        )
        dataframe = pd.concat([dataframe, train_df], ignore_index=True)

    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    if InstructConfig.clean_text:
        dataframe["prompt"] = dataframe["prompt"].apply(cleaner)

    if "rule_violation" in dataframe:
        dataframe["completion"] = dataframe["rule_violation"].map(
            {
                1: InstructConfig.positive_answer,
                0: InstructConfig.negative_answer,
            }
        )

    return dataframe


@DataframeFactory.register(ChatConfig.model_type)
def build_dataframe_chat(dataframe=None, is_train=False):
    def build_prompt(row, tokenizer):
        text = (
            f"\nr/{row['subreddit']}\n"
            f"Rule: {row['rule']}\n"
            "\n"
            f"1) {row['positive_example_1']}\n"
            f"{ChatConfig.complete_phrase} Yes\n\n"
            f"2) {row['positive_example_2']}\n"
            f"{ChatConfig.complete_phrase} Yes\n\n"
            f"3) {row['negative_example_1']}\n"
            f"{ChatConfig.complete_phrase} No\n\n"
            f"4) {row['negative_example_2']}\n"
            f"{ChatConfig.complete_phrase} No\n\n"
            f"5) {row['body']}\n"
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
        return prompt

    tokenizer = AutoTokenizer.from_pretrained(ChatConfig.model_path)
    if is_train or (dataframe is None):  # training with only train.csv
        if not ChatConfig.use_subset:
            dataframe = pd.read_csv(f"{ChatConfig.data_path}/train.csv")
        else:
            dataframe = (
                pd.read_csv(f"{ChatConfig.data_path}/train.csv")
                .sample(frac=ChatConfig.subset, random_state=42)
                .reset_index(drop=True)
            )

    dataframe["prompt"] = dataframe.apply(
        lambda row: build_prompt(row, tokenizer), axis=1
    )

    if ChatConfig.clean_text:
        dataframe["prompt"] = dataframe["prompt"].apply(cleaner)

    if "rule_violation" in dataframe:
        dataframe["completion"] = dataframe["rule_violation"].map(
            {
                1: ChatConfig.positive_answer,
                0: ChatConfig.negative_answer,
            }
        )

    return dataframe


def build_dataframe_emb(dataframe=None):
    def build_prompt(row):
        return f"""r/{row["subreddit"]}\nComment: {row["body"]}"""

    if dataframe is None:  # training
        if not EmbeddingConfig.use_subset:
            dataframe = get_dataframe_to_train_semantic(
                EmbeddingConfig.data_path, EmbeddingConfig.include_train
            )
        else:
            dataframe = get_dataframe_to_train_semantic(
                EmbeddingConfig.data_path,
                EmbeddingConfig.include_train,
                EmbeddingConfig.subset,
            )
    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    if EmbeddingConfig.clean_text:
        dataframe["prompt"] = dataframe["prompt"].apply(cleaner)

    if "rule_violation" in dataframe.columns:
        dataframe["rule_violation"] = dataframe["rule_violation"].map(
            {
                1: EmbeddingConfig.positive_answer,
                0: EmbeddingConfig.negative_answer,
            }
        )

    return dataframe


@DataframeFactory.register(EmbeddingConfig.model_type)
def build_dataframe_emb_swift(is_train=False):
    """
    Swift framework Qwen3 Embedding Fine Tuning
    """
    dataframe = get_dataframe_to_train(EmbeddingConfig.data_path)

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

    if EmbeddingConfig.clean_text:
        dataframe["messages"] = dataframe["messages"].apply(cleaner)

        dataframe["positive_messages"] = dataframe["positive_messages"].apply(
            cleaner
        )

        dataframe["negative_messages"] = dataframe["negative_messages"].apply(
            cleaner
        )

    return build_dataframe_emb(dataframe)


@DataframeFactory.register(RobertaConfig.model_type)
def build_dataframe_roberta(is_train=False):
    def build_prompt(row):
        rule = row["rule"]
        body = row["body"]
        url_features = url_to_semantics(body)

        return f"rule: {rule}\nbody: {body}\n{url_features}"

    if RobertaConfig.include_train:
        train_df = pd.read_csv(f"{RobertaConfig.data_path}/train.csv")
    else:
        train_df = pd.DataFrame()

    if RobertaConfig.test_file is None:
        test_df = pd.read_csv(f"{RobertaConfig.data_path}/test.csv")
    else:
        test_df = pd.read_csv(RobertaConfig.test_file)

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

    if RobertaConfig.include_train:
        train_df = pd.DataFrame(
            train_df[["body", "rule", "subreddit", "rule_violation"]].copy()
        )

    train_df = pd.concat([train_df, df_add], axis=0)

    train_df["input"] = train_df.apply(build_prompt, axis=1)

    test_df["input"] = test_df.apply(build_prompt, axis=1)

    if RobertaConfig.clean_text:
        train_df["input"] = train_df["input"].apply(cleaner)
        test_df["input"] = test_df["input"].apply(cleaner)

    if RobertaConfig.use_subset:
        train_df = train_df.sample(
            frac=RobertaConfig.subset, random_state=42
        ).reset_index(drop=True)
    return train_df, test_df


@DataframeFactory.register(E5Config.model_type)
def build_dataframe_e5(dataframe=None, is_train=False):
    def build_prompt(row):
        return (
            "rule:"
            + row["rule"]
            + "subreddit:"
            + row["subreddit"]
            + "body:"
            + row["body"]
        )

    if dataframe is None:  # training
        if not E5Config.use_subset:
            dataframe = get_dataframe_to_train(E5Config.data_path)
        else:
            dataframe = get_dataframe_to_train(
                E5Config.data_path, E5Config.subset
            )

    dataframe["anchor"] = dataframe.apply(build_prompt, axis=1)
    if E5Config.clean_text:
        dataframe["anchor"] = dataframe["anchor"].apply(cleaner)
    return dataframe


@DataframeFactory.register(BgeConfig.model_type)
def build_dataframe_bge(dataframe=None, is_train=False):
    test_df = pd.read_csv(f"{BgeConfig.data_path}/test.csv")

    random.seed(BgeConfig.RANDOM_STATE)
    np.random.seed(BgeConfig.RANDOM_STATE)

    anchors = []
    positives = []
    negatives = []

    for _, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Processing test rows"
    ):
        rule = cleaner(str(row["rule"]))

        pos_examples = []  # Will contain compliant comments (rule-aligned)
        neg_examples = []  # Will contain violating comments (rule-misaligned)

        for neg_col in [
            "negative_example_1",
            "negative_example_2",
        ]:  # Compliant → triplet positive
            if pd.notna(row[neg_col]):
                pos_examples.append(cleaner(str(row[neg_col])))

        for pos_col in [
            "positive_example_1",
            "positive_example_2",
        ]:  # Violating → triplet negative
            if pd.notna(row[pos_col]):
                neg_examples.append(cleaner(str(row[pos_col])))

        for pos_ex in pos_examples:
            for neg_ex in neg_examples:
                anchors.append(rule)
                positives.append(pos_ex)
                negatives.append(neg_ex)

    if BgeConfig.augmentation_factor > 0:
        rule_positives = {}
        rule_negatives = {}

        for rule in test_df["rule"].unique():
            rule_df = test_df[test_df["rule"] == rule]

            pos_pool = []
            neg_pool = []

            for _, row in rule_df.iterrows():
                for neg_col in [
                    "negative_example_1",
                    "negative_example_2",
                ]:  # Compliant → triplet positive
                    if pd.notna(row[neg_col]):
                        pos_pool.append(cleaner(str(row[neg_col])))
                for pos_col in [
                    "positive_example_1",
                    "positive_example_2",
                ]:  # Violating → triplet negative
                    if pd.notna(row[pos_col]):
                        neg_pool.append(cleaner(str(row[pos_col])))

            rule_positives[rule] = list(set(pos_pool))
            rule_negatives[rule] = list(set(neg_pool))

        for rule in test_df["rule"].unique():
            clean_rule = cleaner(str(rule))
            pos_pool = rule_positives[rule]
            neg_pool = rule_negatives[rule]

            n_samples = min(
                BgeConfig.augmentation_factor * len(pos_pool),
                len(pos_pool) * len(neg_pool),
            )

            for _ in range(n_samples):
                if pos_pool and neg_pool:
                    anchors.append(clean_rule)
                    positives.append(random.choice(pos_pool))
                    negatives.append(random.choice(neg_pool))

    combined = list(zip(anchors, positives, negatives))
    random.shuffle(combined)

    if BgeConfig.subset < 1.0:
        n_samples = int(len(combined) * BgeConfig.subset)
        combined = combined[:n_samples]

    anchors, positives, negatives = (
        zip(*combined) if combined else ([], [], [])
    )

    dataframe = pd.DataFrame.from_dict(
        {
            "anchor": list(anchors),
            "positive": list(positives),
            "negative": list(negatives),
        }
    )

    return dataframe


@DataframeFactory.register(DebertaConfig.model_type)
def build_dataframe_deberta(dataframe=None, is_train=False):
    def build_prompt(row):
        rule = row["rule"]
        body = row["body"]
        url_features = url_to_semantics(body)

        return f"{rule}[SEP]{body}{url_features}"

    if dataframe is None:  # training
        if not DebertaConfig.use_subset:
            dataframe = get_dataframe_to_train(
                DebertaConfig.data_path,
            )
        else:
            dataframe = get_dataframe_to_train(
                DebertaConfig.data_path,
                DebertaConfig.subset,
            )

    if is_train and DebertaConfig.include_train:
        flatten = []
        train_df = pd.read_csv(f"{DebertaConfig.data_path}/train.csv")
        for violation_type in ["positive", "negative"]:
            for i in range(1, 3):
                col_name = f"{violation_type}_example_{i}"

                if col_name in train_df.columns:
                    sub_dataset = train_df[
                        [col_name, "rule", "subreddit"]
                    ].copy()
                    sub_dataset = sub_dataset.rename(
                        columns={col_name: "body"}
                    )
                    sub_dataset["rule_violation"] = (
                        1 if violation_type == "positive" else 0
                    )

                    sub_dataset.dropna(subset=["body"], inplace=True)
                    sub_dataset = sub_dataset[
                        sub_dataset["body"].str.strip().str.len() > 0
                    ]

                    if not sub_dataset.empty:
                        flatten.append(sub_dataset)

        flatten.append(dataframe)
        dataframe = pd.concat(flatten, ignore_index=True)

    dataframe = dataframe.copy()
    dataframe["input_text"] = dataframe.apply(build_prompt, axis=1)

    if DebertaConfig.clean_text:
        dataframe["input_text"] = dataframe["input_text"].apply(cleaner)

    if "rule_violation" in dataframe:
        dataframe["completion"] = dataframe["rule_violation"].map(
            {
                1: DebertaConfig.positive_answer,
                0: DebertaConfig.negative_answer,
            }
        )

    return dataframe


def get_train_dataframe(model_type):
    return DataframeFactory.build(model_type, is_train=True)
