"""
Designed for inference with semantic search
"""

import numpy as np
import pandas as pd  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from jigsaw_rules.configs import BgeConfig, E5Config, EmbeddingConfig
from jigsaw_rules.inference import JigsawInference
from jigsaw_rules.utils import (
    build_dataframe_e5,
    build_dataframe_emb,
    cleaner,
    get_train_dataframe,
)


class Qwen3EmbEngine(JigsawInference):
    def get_dataset(self):
        if EmbeddingConfig.test_file is None:
            dataframe = pd.read_csv(f"{self.data_path}/test.csv")
        else:
            dataframe = pd.read_csv(EmbeddingConfig.test_file)
        dataframe = build_dataframe_emb(dataframe)
        return dataframe

    def get_scores(self, test_dataframe, return_preds=False):
        from peft import PeftConfig, PeftModel
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import dot_score, semantic_search
        from transformers import AutoModelForCausalLM, AutoTokenizer

        corpus_dataframe = get_train_dataframe(EmbeddingConfig.model_type)

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Load adapter configuration and model
        adapter_config = PeftConfig.from_pretrained(self.lora_path)
        lora_model = PeftModel.from_pretrained(
            model, self.lora_path, config=adapter_config
        )
        merged_model = lora_model.merge_and_unload()  # type: ignore
        tokenizer.save_pretrained("Qwen3Emb_Finetuned")
        merged_model.save_pretrained("Qwen3Emb_Finetuned")

        embedding_model = SentenceTransformer(
            model_name_or_path="Qwen3Emb_Finetuned", device="cuda"
        )

        print("Done loading model!")

        result = []
        for rule in tqdm(
            test_dataframe["rule"].unique(),
            desc="Generate scores for each rule",
        ):
            test_dataframe_part = test_dataframe.query(
                "rule == @rule"
            ).reset_index(drop=True)
            corpus_dataframe_part = corpus_dataframe.query(
                "rule == @rule"
            ).reset_index(drop=True)
            corpus_dataframe_part = corpus_dataframe_part.reset_index(
                names="row_id"
            )

            query_embeddings = embedding_model.encode(
                sentences=test_dataframe_part["prompt"].tolist(),
                prompt=EmbeddingConfig.base_query,
                batch_size=EmbeddingConfig.batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                device="cuda",
                normalize_embeddings=True,
            )
            document_embeddings = embedding_model.encode(
                sentences=corpus_dataframe_part["prompt"].tolist(),
                batch_size=EmbeddingConfig.batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                device="cuda",
                normalize_embeddings=True,
            )
            test_dataframe_part["semantic"] = semantic_search(
                query_embeddings,
                document_embeddings,
                top_k=EmbeddingConfig.top_k,
                score_function=dot_score,
            )

            def get_score(semantic):
                semantic = pd.DataFrame(semantic)
                semantic = semantic.merge(
                    corpus_dataframe_part[["row_id", "rule_violation"]],
                    how="left",
                    left_on="corpus_id",
                    right_on="row_id",
                )
                semantic["score"] = (
                    semantic["score"] * semantic["rule_violation"]
                )
                return semantic["score"].sum()

            test_dataframe_part["rule_violation"] = test_dataframe_part[
                "semantic"
            ].apply(get_score)
            result.append(
                test_dataframe_part[["row_id", "rule_violation"]].copy()
            )

        submission = pd.concat(result, axis=0)
        submission = test_dataframe[["row_id"]].merge(
            submission, on="row_id", how="left"
        )

        if return_preds:
            return submission[["rule_violation"]]

        return submission

    def run(self):
        dataframe = self.get_dataset()

        submission = self.get_scores(dataframe)

        submission.to_csv(self.save_path, index=False)


class E5BaseEngine(JigsawInference):
    def get_dataset(self):
        if E5Config.test_file is None:
            dataframe = pd.read_csv(f"{self.data_path}/test.csv")
        else:
            dataframe = pd.read_csv(E5Config.test_file)
        dataframe = build_dataframe_e5(dataframe)
        return dataframe

    def get_scores(self, test_dataframe, return_preds=False):
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import dot_score, semantic_search

        corpus_dataframe = get_train_dataframe(E5Config.model_type)

        embedding_model = SentenceTransformer(
            model_name_or_path=self.model_path, device="cuda"
        )

        print("Done loading model!")

        result = []
        for rule in tqdm(
            test_dataframe["rule"].unique(),
            desc="Generate scores for each rule",
        ):
            test_dataframe_part = test_dataframe.query(
                "rule == @rule"
            ).reset_index(drop=True)
            corpus_dataframe_part = corpus_dataframe.query(
                "rule == @rule"
            ).reset_index(drop=True)
            corpus_dataframe_part = corpus_dataframe_part.reset_index(
                names="row_id"
            )

            query_embeddings = embedding_model.encode(
                sentences=test_dataframe_part["anchor"].tolist(),
                prompt="query: ",
                batch_size=E5Config.batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                device="cuda",
                normalize_embeddings=True,
            )
            document_embeddings = embedding_model.encode(
                sentences=corpus_dataframe_part["anchor"].tolist(),
                prompt="passage: ",
                batch_size=E5Config.batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                device="cuda",
                normalize_embeddings=True,
            )
            test_dataframe_part["semantic"] = semantic_search(
                query_embeddings,
                document_embeddings,
                top_k=E5Config.top_k,
                score_function=dot_score,
            )

            def get_score(semantic):
                semantic = pd.DataFrame(semantic)
                semantic = semantic.merge(
                    corpus_dataframe_part[["row_id", "rule_violation"]],
                    how="left",
                    left_on="corpus_id",
                    right_on="row_id",
                )
                semantic["score"] = (
                    semantic["score"] * semantic["rule_violation"]
                )
                return semantic["score"].sum()

            test_dataframe_part["rule_violation"] = test_dataframe_part[
                "semantic"
            ].apply(get_score)
            result.append(
                test_dataframe_part[["row_id", "rule_violation"]].copy()
            )

        submission = pd.concat(result, axis=0)
        submission = test_dataframe[["row_id"]].merge(
            submission, on="row_id", how="left"
        )

        if return_preds:
            return submission[["rule_violation"]]

        return submission

    def run(self):
        dataframe = self.get_dataset()

        submission = self.get_scores(dataframe)
        submission.to_csv(self.save_path, index=False)


class BgeBaseEngine(JigsawInference):
    def get_dataset(self):
        test_df = pd.read_csv(f"{BgeConfig.data_path}/test.csv")

        return test_df

    def get_scores(self, test_df, return_preds=False):
        from sentence_transformers import SentenceTransformer, models
        from sentence_transformers.util import dot_score, semantic_search

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

        all_texts = set()

        # Add all bodies
        for body in test_df["body"]:
            if pd.notna(body):
                all_texts.add(cleaner(str(body)))

        # Add all positive and negative examples
        example_cols = [
            "positive_example_1",
            "positive_example_2",
            "negative_example_1",
            "negative_example_2",
        ]

        for col in example_cols:
            for example in test_df[col]:
                if pd.notna(example):
                    all_texts.add(cleaner(str(example)))

        all_texts = list(all_texts)

        text_embeddings = model.encode(
            sentences=all_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

        text_to_embedding = {
            text: emb for text, emb in zip(all_texts, text_embeddings)
        }

        unique_rules = test_df["rule"].unique()

        rule_embeddings = {}

        for rule in unique_rules:
            clean_rule = cleaner(str(rule))
            rule_emb = model.encode(
                clean_rule, convert_to_tensor=False, normalize_embeddings=True
            )
            rule_embeddings[rule] = rule_emb

        rule_centroids = {}

        for rule in unique_rules:
            rule_data = test_df[test_df["rule"] == rule]

            # Collect positive examples
            pos_embeddings = []
            for _, row in rule_data.iterrows():
                for col in ["positive_example_1", "positive_example_2"]:
                    if pd.notna(row[col]):
                        clean_text = cleaner(str(row[col]))
                        if clean_text in text_to_embedding:
                            pos_embeddings.append(
                                text_to_embedding[clean_text]
                            )

            # Collect negative examples
            neg_embeddings = []
            for _, row in rule_data.iterrows():
                for col in ["negative_example_1", "negative_example_2"]:
                    if pd.notna(row[col]):
                        clean_text = cleaner(str(row[col]))
                        if clean_text in text_to_embedding:
                            neg_embeddings.append(
                                text_to_embedding[clean_text]
                            )

            if pos_embeddings and neg_embeddings:
                pos_embeddings = np.array(pos_embeddings)
                neg_embeddings = np.array(neg_embeddings)

                # Compute mean centroids
                pos_centroid = pos_embeddings.mean(axis=0)
                neg_centroid = neg_embeddings.mean(axis=0)

                # Normalize centroids
                pos_centroid = pos_centroid / np.linalg.norm(pos_centroid)
                neg_centroid = neg_centroid / np.linalg.norm(neg_centroid)

                rule_centroids[rule] = {
                    "positive": pos_centroid,
                    "negative": neg_centroid,
                    "pos_count": len(pos_embeddings),
                    "neg_count": len(neg_embeddings),
                    "rule_embedding": rule_embeddings[rule],
                }

        row_ids = []
        predictions = []

        for rule in unique_rules:
            print(f"  Processing rule: {rule[:50]}...")
            rule_data = test_df[test_df["rule"] == rule]

            if rule not in rule_centroids:
                continue

            pos_centroid = rule_centroids[rule]["positive"]
            neg_centroid = rule_centroids[rule]["negative"]

            # Collect all valid embeddings and row_ids for this rule
            valid_embeddings = []
            valid_row_ids = []

            for _, row in rule_data.iterrows():
                body = cleaner(str(row["body"]))
                row_id = row["row_id"]

                if body in text_to_embedding:
                    valid_embeddings.append(text_to_embedding[body])
                    valid_row_ids.append(row_id)

            if not valid_embeddings:
                continue

            # Convert to numpy array
            query_embeddings = np.array(valid_embeddings)

            # Compute Euclidean distances
            pos_distances = np.linalg.norm(
                query_embeddings - pos_centroid, axis=1
            )
            neg_distances = np.linalg.norm(
                query_embeddings - neg_centroid, axis=1
            )

            # Score: closer to positive (lower distance) = higher violation score
            rule_predictions = neg_distances - pos_distances

            row_ids.extend(valid_row_ids)
            predictions.extend(rule_predictions)

        submission = pd.DataFrame(
            {"row_id": row_ids, "rule_violation": np.array(predictions)}
        )

        if return_preds:
            return submission[["rule_violation"]]

        return submission

    def run(self):
        dataframe = self.get_dataset()

        submission = self.get_scores(dataframe)
        submission.to_csv(self.save_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)

    args = parser.parse_args()

    if args.type == EmbeddingConfig.model_type:
        inference: JigsawInference = Qwen3EmbEngine(
            data_path=EmbeddingConfig.data_path,
            model_path=EmbeddingConfig.model_path,
            lora_path=EmbeddingConfig.lora_path,
            save_path=EmbeddingConfig.out_file,
        )
        inference.run()
    elif args.type == E5Config.model_type:
        inference = E5BaseEngine(
            data_path=E5Config.data_path,
            model_path=E5Config.ckpt_path,
            save_path=E5Config.out_file,
        )
        inference.run()
    elif args.type == BgeConfig.model_type:
        inference = BgeBaseEngine(
            data_path=BgeConfig.data_path,
            model_path=BgeConfig.ckpt_path,
            save_path=BgeConfig.out_file,
        )
        inference.run()
    else:
        raise AttributeError("Invalid inference type")
