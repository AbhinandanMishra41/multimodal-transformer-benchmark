"""Toy dataset + training loop.

There's no real image-caption data here on purpose: the point of training is
to get weights that aren't random before exporting / quantizing, and to check
that gradients actually flow from the caption loss back into the ViT.

Each image is a coloured square or circle in one of the four quadrants, and
the caption describes it, e.g. "a red circle in the top left".
"""

from dataclasses import dataclass, field
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .model import MultimodalTransformer

PAD, BOS, EOS = 0, 1, 2
WORDS = ["a", "red", "green", "blue", "square", "circle", "in", "the", "top", "bottom", "left", "right"]
WORD_TO_ID = {w: i + 4 for i, w in enumerate(WORDS)}  # ids 0-3 are special tokens
ID_TO_WORD = {i: w for w, i in WORD_TO_ID.items()}

COLORS = ["red", "green", "blue"]  # index == RGB channel
SHAPES = ["square", "circle"]
QUADRANTS = [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]


def encode_caption(text: str) -> list[int]:
    return [BOS] + [WORD_TO_ID[w] for w in text.split()] + [EOS]


def decode_caption(ids) -> str:
    words = []
    for i in ids:
        i = int(i)
        if i == EOS:
            break
        if i in ID_TO_WORD:
            words.append(ID_TO_WORD[i])
        elif i not in (BOS, PAD):
            words.append(f"<{i}>")
    return " ".join(words)


def draw_shape(image_size: int, color: int, shape: int, quadrant: int, noise: torch.Tensor | None = None):
    img = torch.zeros(3, image_size, image_size)
    h = image_size // 2
    y0 = 0 if quadrant in (0, 1) else h
    x0 = 0 if quadrant in (0, 2) else h
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(h), indexing="ij")
    c = (h - 1) / 2
    if SHAPES[shape] == "square":
        m = h // 6
        mask = (yy >= m) & (yy < h - m) & (xx >= m) & (xx < h - m)
    else:
        mask = (yy - c) ** 2 + (xx - c) ** 2 <= (0.4 * h) ** 2
    img[color, y0 : y0 + h, x0 : x0 + h] = mask.float()
    if noise is not None:
        img = img + noise
    return img


class SyntheticMultimodalDataset(Dataset):
    def __init__(self, num_samples: int = 256, image_size: int = 224, seq_len: int = 16, seed: int = 42) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.seq_len = seq_len

        g = torch.Generator().manual_seed(seed)
        self.images = torch.zeros(num_samples, 3, image_size, image_size)
        self.input_ids = torch.full((num_samples, seq_len), PAD, dtype=torch.long)
        self.labels = torch.full((num_samples, seq_len), PAD, dtype=torch.long)
        self.captions: list[str] = []

        for i in range(num_samples):
            color, shape, quad = (int(torch.randint(n, (1,), generator=g)) for n in (3, 2, 4))
            noise = torch.randn(3, image_size, image_size, generator=g) * 0.05
            self.images[i] = draw_shape(image_size, color, shape, quad, noise)

            v, h = QUADRANTS[quad]
            text = f"a {COLORS[color]} {SHAPES[shape]} in the {v} {h}"
            self.captions.append(text)
            ids = torch.tensor(encode_caption(text))
            n = min(len(ids) - 1, seq_len)
            self.input_ids[i, :n] = ids[:n]
            self.labels[i, :n] = ids[1 : n + 1]  # next-token targets, PAD is ignored by the loss

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.input_ids[idx], self.labels[idx]


@dataclass
class TrainingMetrics:
    steps: int
    initial_loss: float
    final_loss: float
    elapsed_sec: float
    samples_per_sec: float
    loss_history: list[float] = field(default_factory=list, repr=False)


def train_synthetic_vlm(
    model: MultimodalTransformer,
    steps: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    use_amp: bool = True,
    seed: int = 0,
    log_every: int = 0,
    grad_accum_steps: int = 1,
) -> TrainingMetrics:
    """Each step sees batch_size * grad_accum_steps samples. Accumulation is there so
    the compact model can train with an effective batch of 32 on a 4GB card."""
    device = torch.device(device)
    model.to(device).train()

    # only 24 distinct captions, so a few hundred noisy samples is plenty and the loader
    # just cycles. Bigger would cost a lot of RAM at 224x224 (512 images ~ 300 MB).
    n = min(max(steps * batch_size * grad_accum_steps, 64), 512)
    ds = SyntheticMultimodalDataset(num_samples=n, image_size=model.config.vision.image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    losses = []
    t0 = time.perf_counter()
    batches = iter(loader)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                images, input_ids, labels = next(batches)
            except StopIteration:
                batches = iter(loader)
                images, input_ids, labels = next(batches)
            images, input_ids, labels = images.to(device), input_ids.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _, loss = model(images, input_ids, labels=labels)
            scaler.scale(loss / grad_accum_steps).backward()
            step_loss += loss.item() / grad_accum_steps
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        losses.append(step_loss)
        if log_every and step % log_every == 0:
            print(f"step {step:4d}  loss {losses[-1]:.3f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    model.eval()
    # average the first/last few steps, single-step losses are noisy
    k = min(3, len(losses))
    return TrainingMetrics(
        steps=steps,
        initial_loss=sum(losses[:k]) / k,
        final_loss=sum(losses[-k:]) / k,
        elapsed_sec=elapsed,
        samples_per_sec=steps * batch_size * grad_accum_steps / elapsed,
        loss_history=losses,
    )


@torch.inference_mode()
def caption_accuracy(model: MultimodalTransformer, num_samples: int = 64, seed: int = 1234) -> float:
    """Fraction of freshly drawn images where the greedy caption matches exactly.

    These are new samples (different seed, so different noise), but they come from
    the same 24 colour/shape/position combinations as training - it checks that
    the model reads the image, not that it generalises."""
    p = next(model.parameters())
    ds = SyntheticMultimodalDataset(num_samples=num_samples, image_size=model.config.vision.image_size, seed=seed)
    hits = 0
    for i in range(0, num_samples, 16):  # chunks, so 224px images fit on a small GPU
        images = ds.images[i : i + 16].to(p.device, p.dtype)
        prompt = torch.full((images.shape[0], 1), BOS, dtype=torch.long, device=p.device)
        out = model.generate(images, prompt, max_new_tokens=10)
        hits += sum(decode_caption(row[1:]) == cap for row, cap in zip(out.tolist(), ds.captions[i : i + 16]))
    return hits / num_samples
