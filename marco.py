import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import glob
import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import random

# Set up argument parser
parser = argparse.ArgumentParser(
    description="MS MARCO Two-Tower Model with BERT Embeddings"
)
parser.add_argument("--train", action="store_true", help="Run training")
parser.add_argument(
    "--optimize", action="store_true", help="Run hyperparameter optimization"
)
parser.add_argument("--bs", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--outdim", type=int, default=128)
parser.add_argument("--arch", type=str)
args = parser.parse_args()

# Constants
MAX_PASSAGES = 11
INPUT_SIZE = 768
PREPROCESSED_DIR = "preprocessed_ms_marco"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


class Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(INPUT_SIZE, args.outdim)

    def forward(self, embeds):
        outputs = self.fc0(embeds)
        return outputs

    def reg_loss(self):
        return 0


class LRTower(nn.Module):
    def __init__(self, lowrank_dim=128):
        super().__init__()
        self.fc0 = nn.Linear(INPUT_SIZE, lowrank_dim)
        self.fc1 = nn.Linear(lowrank_dim, args.outdim)
        assert INPUT_SIZE == args.outdim
        self.diag = nn.Parameter(torch.ones(args.outdim))

    def forward(self, embeds):
        outputs = self.fc1(self.fc0(embeds))
        return outputs + self.diag * embeds

    def reg_loss(self):
        return 0


class LRTowerSimple(nn.Module):
    def __init__(self, lowrank_dim=128):
        super().__init__()
        self.fc0 = nn.Linear(INPUT_SIZE, lowrank_dim)
        self.fc1 = nn.Linear(lowrank_dim, args.outdim)

    def forward(self, embeds):
        outputs = self.fc1(self.fc0(embeds))
        return outputs

    def reg_loss(self):
        return 0


class LRTowerProb(nn.Module):
    def __init__(self, lowrank_dim=128):
        super().__init__()
        self.fc0 = nn.Linear(INPUT_SIZE, lowrank_dim)
        self.fc1 = nn.Linear(lowrank_dim, args.outdim)
        self.sigma = nn.Parameter(torch.zeros(lowrank_dim))

    def forward(self, embeds):
        outputs = self.fc0(embeds)

        if self.training:
            outputs = outputs + torch.randn_like(outputs) * torch.exp(self.sigma)

        return self.fc1(outputs)

    def reg_loss(self):
        return -torch.mean(self.sigma)


def contrastive_loss(query_embeds, passage_embeds, labels):
    batch_size, n_passages, dim = passage_embeds.shape
    scores = torch.einsum("bd,bpd->bp", query_embeds, passage_embeds)
    targets = labels.argmax(dim=-1)
    loss = F.cross_entropy(scores, targets)
    return loss


def get_data_files(split="train"):
    query_files = sorted(
        glob.glob(os.path.join(PREPROCESSED_DIR, f"{split}_query_embeddings_*.pt"))
    )
    passage_files = sorted(
        glob.glob(os.path.join(PREPROCESSED_DIR, f"{split}_passage_embeddings_*.pt"))
    )
    label_files = sorted(
        glob.glob(os.path.join(PREPROCESSED_DIR, f"{split}_labels_*.pt"))
    )
    return list(zip(query_files, passage_files, label_files))


def load_batches(file_paths, batch_size):
    for q_file, p_file, l_file in file_paths:
        query_embeds = torch.load(q_file)
        passage_embeds = torch.load(p_file)
        labels = torch.load(l_file)

        # Add dummy to labels
        dummy_labels = (labels.sum(dim=1) == 0).float().unsqueeze(1)
        labels = torch.cat([labels, dummy_labels], dim=1)

        for i in range(0, len(query_embeds), batch_size):
            batch_query = query_embeds[i : i + batch_size].to(device).to(dtype)
            batch_passage = passage_embeds[i : i + batch_size].to(device).to(dtype)
            batch_labels = labels[i : i + batch_size].to(device).to(dtype)

            yield batch_query, batch_passage, batch_labels


def compute_total_batches(file_paths, batch_size):
    total_batches = 0
    for q_file, _, _ in file_paths:
        query_embeds = torch.load(q_file)
        total_batches += (len(query_embeds) + batch_size - 1) // batch_size
    return total_batches


def mrr(scores, labels):
    # Compute ranks of all passages
    ranks = (-scores).argsort(dim=1) + 1
    
    # Find the rank of the relevant document (if it exists)
    relevant_ranks = (ranks * labels).sum(dim=1)
    
    # Compute MRR only for queries with a relevant document
    mrr = (1.0 / relevant_ranks.float()).mean().item()
    
    return mrr

def evaluate(tower1, tower2, val_files, batch_size):
    tower1.eval()
    tower2.eval()
    total_correct = 0
    total_samples = 0
    total_mrr = 0

    total_batches = compute_total_batches(val_files, batch_size)

    with torch.no_grad():
        for query_embeds, passage_embeds, labels in tqdm(
            load_batches(val_files, batch_size), total=total_batches, desc="Evaluating"
        ):
            query_outputs = tower1(query_embeds)
            passage_outputs = tower2(passage_embeds)
            scores = torch.einsum("bd,bpd->bp", query_outputs, passage_outputs)

            predicted = scores.argmax(dim=-1)
            correct = (predicted == labels.argmax(dim=-1)).sum().item()
            total_correct += correct
            total_samples += len(query_embeds)

            mrr_at_10 = mrr(scores, labels)
            total_mrr += mrr_at_10 * len(query_embeds)

    accuracy = total_correct / total_samples
    avg_mrr = total_mrr / total_samples

    return accuracy, avg_mrr


def train(config):
    if args.arch == "linear":
        tower1 = Tower().to(device).to(dtype)
        tower2 = Tower().to(device).to(dtype)
    elif args.arch == "lowrank+diag":
        tower1 = LRTower().to(device).to(dtype)
        tower2 = LRTower().to(device).to(dtype)
    elif args.arch == "lowrank":
        tower1 = LRTowerSimple().to(device).to(dtype)
        tower2 = LRTowerSimple().to(device).to(dtype)
    elif args.arch == "lowrank+prob":
        tower1 = LRTowerProb().to(device).to(dtype)
        tower2 = LRTowerProb().to(device).to(dtype)
    optimizer = optim.Adam(
        list(tower1.parameters()) + list(tower2.parameters()),
        lr=config["learning_rate"],
    )

    train_files = get_data_files(split="train")
    val_files = get_data_files(split="validation")

    for epoch in range(config["epochs"]):
        tower1.train()
        tower2.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # Shuffle training files for each epoch
        random.shuffle(train_files)

        total_batches = compute_total_batches(train_files, config["batch_size"])
        progress_bar = tqdm(
            load_batches(train_files, config["batch_size"]),
            total=total_batches,
            desc=f"Epoch {epoch+1}/{config['epochs']}",
        )

        for batch_idx, (query_embeds, passage_embeds, labels) in enumerate(
            progress_bar
        ):
            optimizer.zero_grad()

            query_outputs = tower1(query_embeds)
            passage_outputs = tower2(passage_embeds)
            loss = contrastive_loss(query_outputs, passage_outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += tower1.reg_loss() + tower2.reg_loss()

            # Compute accuracy
            scores = torch.einsum("bd,bpd->bp", query_outputs, passage_outputs)
            predicted = scores.argmax(dim=-1)
            correct = (predicted == labels.argmax(dim=-1)).sum().item()
            total_correct += correct
            total_samples += len(query_embeds)

            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = total_correct / total_samples
            progress_bar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{accuracy:.4f}",
                    "lr": f'{optimizer.param_groups[0]["lr"]:.2e}',
                }
            )

            # Log metrics to wandb
            wandb.log(
                {
                    "batch": epoch * total_batches + batch_idx,
                    "loss": loss.item(),
                    "accuracy": correct / len(query_embeds),
                }
            )

        # Evaluate on the validation dataset
        accuracy, mrr_at_10 = evaluate(tower1, tower2, val_files, config["batch_size"])

        # Log epoch metrics to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "val_accuracy": accuracy,
                "val_mrr_at_10": mrr_at_10,
            }
        )

        print(
            f"Epoch {epoch+1}/{config['epochs']}, Avg Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}, Val MRR@10: {mrr_at_10:.4f}"
        )

    return accuracy, mrr_at_10


def objective(trial):
    wandb.init(project="ms-marco-two-tower", group="optimization")

    config = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "epochs": 5,  # Fixed number of epochs for quicker trials
    }

    wandb.config.update(config)

    accuracy, mrr_at_10 = train(config)

    wandb.finish()

    return mrr_at_10  # We'll optimize for MRR@10


if __name__ == "__main__":
    if args.train:
        wandb.init(
            project="ms-marco-two-tower",
            config={
                "learning_rate": args.lr,
                "architecture": args.arch,
                "dataset": "MS MARCO",
                "epochs": args.epochs,
                "batch_size": args.bs,
            },
        )
        train(wandb.config)
        wandb.finish()

    elif args.optimize:
        wandb_callback = WeightsAndBiasesCallback(
            metric_name="mrr_at_10", wandb_kwargs={"project": "ms-marco-two-tower"}
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, callbacks=[wandb_callback])

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
