# Base Model Configurations
BASE_MODEL_PATH = (
    "/kaggle/input/qwen2.5/transformers/0.5b-instruct-gptq-int4/1"
)
LORA_PATH = "output/"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
BASE_PROMPT = """You are given a comment from reddit and a rule. Your task is to classify whether the comment violates the rule. Only respond Yes/No."""


# Embedding Model Configuration
EMBEDDING_MODEL_PATH = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
EMBEDDING_MODEL_OUTPUT_PATH = "/kaggle/input/qwen3-8b-embedding"

# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config_sentence_transformers.json
EMBEDDING_MODEL_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

CLEAN_TEXT = True
TOP_K = 2000
BATCH_SIZE = 128
