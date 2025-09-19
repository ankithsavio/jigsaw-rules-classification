import pandas as pd  # type: ignore
from peft import PeftConfig, PeftModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score, semantic_search
from tqdm.auto import tqdm  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer

from jigsaw_rules.constants import EmbeddingConfig
from jigsaw_rules.inference import JigsawInference
from jigsaw_rules.utils import build_dataset_emb, get_train_dataset


class EmbeddingEngine(JigsawInference):
    def get_dataset(self):
        dataframe = pd.read_csv(f"{self.data_path}/test.csv")
        dataframe = build_dataset_emb(dataframe)
        return dataframe

    def get_scores(self, test_dataframe):
        corpus_dataframe = get_train_dataset("embed")

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

        # 4. Tạo lại SentenceTransformer từ encoder đã merge
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

            tqdm.pandas(desc=f"Add label for {rule=}")
            test_dataframe_part["rule_violation"] = test_dataframe_part[
                "semantic"
            ].progress_apply(get_score)
            result.append(
                test_dataframe_part[["row_id", "rule_violation"]].copy()
            )

        submission = pd.concat(result, axis=0)
        return submission

    def run(self):
        dataframe = self.get_dataset()

        submission = self.get_scores(dataframe)
        submission = dataframe[["row_id"]].merge(
            submission, on="row_id", how="left"
        )
        submission.to_csv(self.save_path, index=False)


if __name__ == "__main__":
    inference = EmbeddingEngine(
        data_path=EmbeddingConfig.data_path,
        model_path=EmbeddingConfig.model_path,
        lora_path=EmbeddingConfig.lora_path,
        save_path=EmbeddingConfig.out_file,
    )

    inference.run()
