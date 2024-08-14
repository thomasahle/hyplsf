import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from tqdm import tqdm
import os

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained(
    "bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa"
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)


class MSMARCODataset(Dataset):
    def __init__(self, dataset, split="train", max_passages=8):
        self.dataset = dataset[split]
        self.max_passages = max_passages

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        query = item["query"]
        passages = item["passages"]["passage_text"][: self.max_passages]
        labels = item["passages"]["is_selected"][: self.max_passages]

        # Pad passages and labels if necessary
        passages = passages + [""] * (self.max_passages - len(passages))
        labels = labels + [0] * (self.max_passages - len(labels))

        return query, passages, labels


def collate_fn(batch):
    queries, passages, labels = zip(*batch)

    # Tokenize queries
    query_inputs = tokenizer(
        list(queries), padding=True, truncation=True, return_tensors="pt"
    )

    # Tokenize passages
    flat_passages = [p for sublist in passages for p in sublist]
    passage_inputs = tokenizer(
        flat_passages, padding=True, truncation=True, return_tensors="pt"
    )

    # Reshape passage inputs
    batch_size = len(queries)
    max_passages = len(passages[0])
    for key in passage_inputs:
        passage_inputs[key] = passage_inputs[key].view(batch_size, max_passages, -1)

    # Convert labels to tensor
    labels_tensor = torch.tensor(labels)

    return query_inputs, passage_inputs, labels_tensor


def preprocess_dataset(dataset, split, output_dir, batch_size=2048, save_every=100):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    os.makedirs(output_dir, exist_ok=True)

    query_embeddings = []
    passage_embeddings = []
    labels_list = []

    with torch.no_grad():
        for i, (query_inputs, passage_inputs, labels) in enumerate(
            tqdm(data_loader, desc=f"Processing {split} set")
        ):
            # Move inputs to device
            query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
            passage_inputs = {k: v.to(device) for k, v in passage_inputs.items()}

            # Get query embeddings
            query_outputs = bert_model(**query_inputs)[1]  # Use pooler output
            query_embeddings.append(query_outputs.cpu())

            # Get passage embeddings
            batch_size, num_passages, seq_length = passage_inputs["input_ids"].shape
            for k, v in passage_inputs.items():
                passage_inputs[k] = v.view(batch_size * num_passages, seq_length)

            passage_outputs = bert_model(**passage_inputs)[1]
            passage_outputs = passage_outputs.view(batch_size, num_passages, -1)
            passage_embeddings.append(passage_outputs.cpu())

            labels_list.append(labels)

            # Save batches periodically to free up memory
            if (i + 1) % save_every == 0:
                save_batch(
                    query_embeddings,
                    passage_embeddings,
                    labels_list,
                    output_dir,
                    split,
                    i // save_every,
                )
                query_embeddings = []
                passage_embeddings = []
                labels_list = []

    # Save any remaining data
    if query_embeddings:
        save_batch(
            query_embeddings,
            passage_embeddings,
            labels_list,
            output_dir,
            split,
            (i // save_every) + 1,
        )


def save_batch(
    query_embeddings, passage_embeddings, labels_list, output_dir, split, batch_num
):
    query_embeddings = torch.cat(query_embeddings, dim=0)
    passage_embeddings = torch.cat(passage_embeddings, dim=0)
    labels = torch.cat(labels_list, dim=0)

    torch.save(
        query_embeddings,
        os.path.join(output_dir, f"{split}_query_embeddings_{batch_num:04d}.pt"),
    )
    torch.save(
        passage_embeddings,
        os.path.join(output_dir, f"{split}_passage_embeddings_{batch_num:04d}.pt"),
    )
    torch.save(labels, os.path.join(output_dir, f"{split}_labels_{batch_num}.pt"))


# Load MS MARCO dataset
dataset = load_dataset("ms_marco", "v2.1")

# Preprocess train and validation sets
for split in ["train", "validation"]:
    ms_marco_dataset = MSMARCODataset(dataset, split=split)
    preprocess_dataset(ms_marco_dataset, split, "preprocessed_ms_marco")

print("Preprocessing completed!")
