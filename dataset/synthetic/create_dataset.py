import json

import pandas as pd  # type: ignore


def main():
    with open("dataset/synthetic/comments.json", "r") as file:
        comments_json = json.load(file)

    df = (
        pd.DataFrame.from_dict(comments_json)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    rule_df = {rule: df[df["rule"] == rule] for rule in df["rule"].unique()}

    df_collection = []

    for rule in rule_df:
        dataframe = rule_df[rule]

        violating_subset = (
            dataframe[dataframe["rule_violation"] == 1]
            .sample(frac=0.25, random_state=42)
            .reset_index(drop=True)
        )
        non_violating_subset = (
            dataframe[dataframe["rule_violation"] == 0]
            .sample(frac=0.25, random_state=42)
            .reset_index(drop=True)
        )
        test_dataframe = dataframe.copy()
        test_dataframe[["positive_example_1", "positive_example_2"]] = (
            test_dataframe.apply(
                lambda row: violating_subset.loc[
                    violating_subset["body"] != row["body"], "body"
                ]
                .sample(2)
                .tolist(),
                axis=1,
                result_type="expand",
            )
        )

        test_dataframe[["negative_example_1", "negative_example_2"]] = (
            test_dataframe.apply(
                lambda row: non_violating_subset.loc[
                    non_violating_subset["body"] != row["body"], "body"
                ]
                .sample(2)
                .tolist(),
                axis=1,
                result_type="expand",
            )
        )
        test_dataframe = test_dataframe.reset_index(drop=True)
        df_collection.append(test_dataframe)

    dataframe = pd.concat(df_collection, axis=0)
    dataframe = dataframe.drop_duplicates(ignore_index=True)
    dataframe.to_csv("dataset/synthetic/train.csv")


if __name__ == "__main__":
    main()
