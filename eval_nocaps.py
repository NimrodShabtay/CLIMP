import argparse
from contextlib import suppress

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

from models import load_climp
from data.utils import transform_image


class NoCapsDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        image = self.transform(image)
        captions = item["annotations_captions"]
        return {"image": image, "annotations_captions": captions}


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    captions = [item["annotations_captions"] for item in batch]
    return images, captions


def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end


def recall_at_k(scores, positive_pairs, k):
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(
        topk_indices, num_classes=nb_images
    )
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    return nb_true_positive / nb_positive


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def evaluate(model, dataloader, tokenizer, device, amp=True, recall_k_list=[1, 3, 5, 10]):
    batch_images_emb_list = []
    batch_texts_emb_list = []
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.amp.autocast if amp else suppress

    for batch_images, batch_texts, inds in tqdm(dataloader, desc="Encoding"):
        batch_images = batch_images.to(device)
        texts_ = [text for texts in batch_texts for text in texts]
        inputs = tokenizer(
            texts_,
            max_length=max(len(t) for t in texts_),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch_texts_tok = inputs["input_ids"].to(device)
        batch_attn_mask = inputs["attention_mask"].to(device)

        batch_texts_image_index = [
            ind for ind, texts in zip(inds, batch_texts) for text in texts
        ]

        with torch.no_grad(), autocast(device_type="cuda"):
            batch_texts_emb = F.normalize(
                model.encode_text(batch_texts_tok, batch_attn_mask), dim=-1
            )
            batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

    batch_size = len(batch_images_emb_list[0])
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    scores = texts_emb.float() @ images_emb.float().t()

    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True

    metrics = {}
    for k in recall_k_list:
        metrics[f"image_retrieval_recall@{k}"] = (
            (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=k) > 0)
            .float()
            .mean()
            .item()
        )
        metrics[f"text_retrieval_recall@{k}"] = (
            (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=k) > 0)
            .float()
            .mean()
            .item()
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CLIMP NoCaps Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mamba1", "mamba2"],
        help="Model variant: mamba1 or mamba2",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model.safetensors (downloads from HuggingFace if not provided)",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Image resolution")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    print(f"Loading CLIMP-{args.model} (image_size={args.image_size})...")
    model = load_climp(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        device=args.device,
    )
    tokenizer = model.text_tokenizer

    print("Loading NoCaps validation set...")
    nocaps_dataset = load_dataset("HuggingFaceM4/NoCaps", split="validation")
    transform = transform_image(args.image_size)
    dataset = NoCapsDataset(nocaps_dataset, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    print("Running evaluation...")
    metrics = evaluate(model, dataloader, tokenizer, args.device)

    print(f"\n{'='*50}")
    print(f"CLIMP-{args.model} | Image Resolution: {args.image_size}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
