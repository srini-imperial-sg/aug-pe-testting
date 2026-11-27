import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import math
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


from torch.nn import CrossEntropyLoss
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_data_path", type=str, required=True)
    parser.add_argument("--real_data_path", type=str, required=True)
    parser.add_argument("--noise_level", type=float, required=True)
    parser.add_argument("--fold_num", type=str, required=True)
    return parser.parse_args()


def retain_best_perplexity_examples(df_perplexity, df_original):
    label_counts = df_original['labels'].value_counts()
    filtered_df = []
    for label, count in label_counts.items():
        df_perplexity_label = df_perplexity[df_perplexity['label1'] == label]
        top_k = df_perplexity_label.nsmallest(count, 'perplexity')
        filtered_df.append(top_k)
    return pd.concat(filtered_df).reset_index(drop=True)

class Perplexity:
    def __init__(self, model_id, device=None):
        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                self.device = "cuda"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    
    def _compute(
        self, predictions, batch_size: int = 16, add_start_token: bool = True,  max_length=None
    ):
        # special token to also be the padding token
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = self.tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in range(0, len(encoded_texts), batch_size):
            print(start_index)
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(self.device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return ppls #{"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
    
# Load model and tokenizer

if __name__ == "__main__":
    args = get_args()
    model_name = "starmpcc/Asclepius-Llama3-8B"
    perplexity_class = Perplexity(model_name)

    synthetic_data_path = args.synthetic_data_path
    real_data_path = args.real_data_path
    noise_level = args.noise_level
    fold_num = args.fold_num

    df_synthetic = pd.read_csv(synthetic_data_path)
    df_original = pd.read_csv(real_data_path)


    df_synthetic["perplexity"] = perplexity_class._compute(df_synthetic["text"].to_list(), batch_size=2)

    main_dir = f"/home/srini/comparison_study_aug_pe/aug-pe-testting/data/aaai/folds/output_full_data/fold{fold_num}/noise_{noise_level}"
    os.makedirs(main_dir, exist_ok=True)
    
    # save df_synthetic with perplexity column
    df_synthetic.to_csv(f"/home/srini/comparison_study_aug_pe/aug-pe-testting/data/aaai/folds/output_full_data/fold{fold_num}/noise_{noise_level}/synthetic_perplexity.csv", index=False)

    df_synthetic_filtered = retain_best_perplexity_examples(df_synthetic, df_original)

    # save df_synthetic_filtered
    df_synthetic_filtered.to_csv(f"/home/srini/comparison_study_aug_pe/aug-pe-testting/data/aaai/folds/output_full_data/fold{fold_num}/noise_{noise_level}/output.csv", index=False)