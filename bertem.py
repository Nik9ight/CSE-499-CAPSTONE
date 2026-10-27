import os
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from collections import OrderedDict


def _ensure_hf_model_local(model_name: str, lib_dir: str) -> str:
    """Return a local directory holding the HF model snapshot, downloading once if absent.

    Layout: <lib_dir>/<model_basename>/  where <model_basename> is the repo name
    after the org slash (e.g. 'BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext').
    """
    basename = model_name.split('/')[-1]
    local_dir = os.path.join(lib_dir, basename)
    # Treat dir as populated only if a config.json is present
    if not os.path.exists(os.path.join(local_dir, 'config.json')):
        from huggingface_hub import snapshot_download
        os.makedirs(local_dir, exist_ok=True)
        print(f"[lib/] downloading {model_name} -> {local_dir}")
        snapshot_download(repo_id=model_name, local_dir=local_dir)
    return local_dir


class BertEmbeddingWrapper:
    def __init__(self, model_name='microsoft/BiomedVLP-CXR-BERT-specialized', use_cuda=True, lib_dir=None):
        if lib_dir is None:
            import Config as _config
            lib_dir = getattr(_config, 'lib_dir', './lib/')
        local_path = _ensure_hf_model_local(model_name, lib_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(local_path, trust_remote_code=True)
        self.model.eval()

        # device logic
        if use_cuda and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        # self.device = 'cpu'
        print(f"BertEmbeddingWrapper using device: {self.device}")
        self.model.to(self.device)

    def __call__(self, sentences, chunk_size=16):
        results = []
        with torch.no_grad():
            for start in range(0, len(sentences), chunk_size):
                chunk = sentences[start: start + chunk_size]
                inputs = self.tokenizer(
                    chunk,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=80
                ).to(self.device)

                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.last_hidden_state          # [B, T, D]

                for i in range(last_hidden.shape[0]):
                    token_ids = inputs['input_ids'][i].cpu()
                    tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                    token_embeddings = last_hidden[i].cpu().numpy()
                    results.append((tokens, token_embeddings))
        return results


if __name__ == "__main__":
    import os
    import Config as config

    bert_embedding = BertEmbeddingWrapper(model_name=config.text_encoder)

    # Read both splits' prompt CSVs and combine into a single embedding table.
    csv_paths = [config.train_text_csv, config.test_text_csv]
    dfs = []
    for p in csv_paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
        else:
            print(f"Warning: prompt CSV not found, skipping: {p}")
    if not dfs:
        raise FileNotFoundError("No prompt CSVs found; check config.train_text_csv / config.test_text_csv")
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["Image"]).reset_index(drop=True)

    all_texts = df["Description"].astype(str).tolist()
    all_imgs  = df["Image"].astype(str).tolist()

    results = bert_embedding(all_texts)   # [(tokens, emb_np), ...]
    lengths = [emb.shape[0] for _, emb in results]
    max_T = max(lengths)
    D = results[0][1].shape[1]
    N = len(results)

    tensor = torch.zeros((N, max_T, D), dtype=torch.float32)
    lengths_arr = torch.zeros(N, dtype=torch.int32)
    tokens_list = [None] * N
    index_map = {}

    for i, (img, (tokens, emb_np)) in enumerate(zip(all_imgs, results)):
        L = emb_np.shape[0]
        tensor[i, :L, :] = torch.from_numpy(emb_np)
        lengths_arr[i] = L
        tokens_list[i] = tokens
        index_map[img] = i

    os.makedirs("encodings", exist_ok=True)
    torch.save({
        "embeddings": tensor,        # [N, max_T, D]
        "lengths": lengths_arr,      # [N]
        "tokens": tokens_list,       # list of token lists
        "index_map": index_map       # img_name -> row
    }, "encodings/CovOri_anno_enc.pt")
    print("Saved embeddings to encodings/CovOri_anno_enc.pt")