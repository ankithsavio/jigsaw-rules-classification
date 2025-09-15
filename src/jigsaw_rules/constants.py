class InstructConfig:
    model_type = "instruct"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    model_path = "/kaggle/input/qwen2.5/transformers/0.5b-instruct-gptq-int4/1"
    lora_path = "output/"
    positive_answer = "Yes"
    negative_answer = "No"
    complete_phrase = "Answer:"
    base_prompt = """You are given a comment from reddit and a rule. Your task is to classify whether the comment violates the rule. Only respond Yes/No."""
    out_file = "submission_instruct.csv"


class ChatConfig:
    model_type = "chat"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    model_path = "/kaggle/input/qwen2.5/transformers/14b-instruct-gptq-int4/1"
    lora_path = "/kaggle/input/lora_14b_gptq_1epoch_r32/keras/default/1"
    positive_answer = "Yes"
    negative_answer = "No"
    complete_phrase = "Violation:"
    base_prompt = """You are given a comment on reddit. Your task is to classify if it violates the given rule. Only respond Yes/No."""
    out_file = "submission_chat.csv"


class EmbeddingConfig:
    model_type = "embed"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    model_path = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
    lora_path = "/kaggle/input/qwen3-8b-embedding"  # only named as 8b but actually uses 0.6b
    positive_answer = 1
    negative_answer = -1
    base_query = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    clean_text = True
    top_k = 2000
    batch_size = 128
    out_file = "submission_embedding.csv"
