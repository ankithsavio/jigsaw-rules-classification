import pandas as pd
from peft import PeftConfig, PeftModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score, semantic_search
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from jigsaw_rules.constants import (
    BATCH_SIZE,
    DATA_PATH,
    EMBEDDING_MODEL_OUTPUT_PATH,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_MODEL_QUERY,
    TOP_K,
)
from jigsaw_rules.utils import build_dataset_emb, get_dataframe_to_train_emb


def get_scores(test_dataframe):
    corpus_dataframe = get_dataframe_to_train_emb(DATA_PATH)
    corpus_dataframe = build_dataset_emb(corpus_dataframe)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(EMBEDDING_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)

    # Load adapter configuration and model
    adapter_config = PeftConfig.from_pretrained(EMBEDDING_MODEL_OUTPUT_PATH)
    lora_model = PeftModel.from_pretrained(
        model, EMBEDDING_MODEL_OUTPUT_PATH, config=adapter_config
    )
    merged_model = lora_model.merge_and_unload()
    tokenizer.save_pretrained("Qwen3Emb_Finetuned")
    merged_model.save_pretrained("Qwen3Emb_Finetuned")

    # 4. Tạo lại SentenceTransformer từ encoder đã merge
    embedding_model = SentenceTransformer(
        model_name_or_path="Qwen3Emb_Finetuned", device="cuda"
    )

    print("Done loading model!")

    result = []
    for rule in tqdm(
        test_dataframe["rule"].unique(), desc=f"Generate scores for each rule"
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
            prompt=EMBEDDING_MODEL_QUERY,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )
        document_embeddings = embedding_model.encode(
            sentences=corpus_dataframe_part["prompt"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )
        test_dataframe_part["semantic"] = semantic_search(
            query_embeddings,
            document_embeddings,
            top_k=TOP_K,
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
            semantic["score"] = semantic["score"] * semantic["rule_violation"]
            return semantic["score"].sum()

        tqdm.pandas(desc=f"Add label for {rule=}")
        test_dataframe_part["rule_violation"] = test_dataframe_part[
            "semantic"
        ].progress_apply(get_score)
        result.append(test_dataframe_part[["row_id", "rule_violation"]].copy())

    submission = pd.concat(result, axis=0)
    return submission


def generate_submission():
    test_dataframe = pd.read_csv(f"{DATA_PATH}/test.csv")
    test_dataframe = build_dataset_emb(test_dataframe)

    submission = get_scores(test_dataframe)
    submission = test_dataframe[["row_id"]].merge(
        submission, on="row_id", how="left"
    )
    submission.to_csv("submission_qwen3.csv", index=False)


if __name__ == "__main__":
    generate_submission()
