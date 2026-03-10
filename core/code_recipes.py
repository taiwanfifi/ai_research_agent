"""
Code Recipes — Pre-Vetted Code Snippets
=========================================
Eliminates boilerplate errors (CIFAR-10 transforms, VGG dimensions,
training loops) by injecting verified working code into coder context.

Root cause: 84% of LLM code errors need >50 edits or different approach.
Most coder failures are boilerplate, not research logic (ICSE 2025).

Recipes are injected based on keyword matching of task description.
"""

import re

RECIPES = {
    "cifar10_dataloader": '''
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def get_cifar10(batch_size=128, num_samples=None, num_workers=0):
    """CIFAR-10 data loading — verified working.
    Args:
        num_samples: If set, use only first N samples (for fast experiments).
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    if num_samples:
        trainset = Subset(trainset, list(range(min(num_samples, len(trainset)))))
        testset = Subset(testset, list(range(min(num_samples // 5, len(testset)))))
    train_loader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
''',

    "mnist_dataloader": '''
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def get_mnist(batch_size=128, num_samples=None, num_workers=0):
    """MNIST data loading — verified working."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    if num_samples:
        trainset = Subset(trainset, list(range(min(num_samples, len(trainset)))))
        testset = Subset(testset, list(range(min(num_samples // 5, len(testset)))))
    train_loader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
''',

    "sst2_dataloader": '''
from datasets import load_dataset
from transformers import AutoTokenizer

def get_sst2(model_name="distilbert-base-uncased", max_length=128, num_samples=None):
    """SST-2 sentiment data loading — verified working."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("glue", "sst2")

    def tokenize(examples):
        return tokenizer(examples["sentence"], padding="max_length",
                        truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_data = tokenized["train"]
    val_data = tokenized["validation"]
    if num_samples:
        train_data = train_data.select(range(min(num_samples, len(train_data))))
        val_data = val_data.select(range(min(num_samples // 5, len(val_data))))

    return train_data, val_data, tokenizer
''',

    "vgg_cifar": '''
import torch.nn as nn

def get_vgg_cifar(num_classes=10, dropout=0.5):
    """VGG-style CNN for CIFAR-10 — correct FC input size.
    CRITICAL: CIFAR-10 is 32x32. After 5 MaxPool(2,2) → 1x1 spatial.
    FC input = 512 * 1 * 1 = 512 (NOT 4096 like ImageNet VGG!)
    """
    features = nn.Sequential(
        # Block 1
        nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
        nn.MaxPool2d(2, 2),
        # Block 2
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
        nn.MaxPool2d(2, 2),
        # Block 3
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
        nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
        nn.MaxPool2d(2, 2),
        # Block 4
        nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
        nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
        nn.MaxPool2d(2, 2),
        # Block 5
        nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
        nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
        nn.MaxPool2d(2, 2),
    )
    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 512),  # 512, NOT 4096
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )
    return nn.Sequential(features, classifier)
''',

    "simple_cnn_cifar": '''
import torch.nn as nn

def get_simple_cnn(num_classes=10, dropout=0.0):
    """Simple 3-4 layer CNN for CIFAR-10 — verified working.
    Input: 3x32x32 → Output: num_classes
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )
''',

    "training_loop": '''
import torch

def train_epoch(model, loader, criterion, optimizer, device):
    """Standard training loop — verified working."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, 100. * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Standard evaluation — with no_grad and eval mode."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, 100. * correct / total
''',

    "multi_seed_eval": '''
import json, torch, random, numpy as np

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_multi_seed(train_fn, seeds=[42, 123, 456, 789, 1024], save_path="results.json"):
    """Multi-seed evaluation with JSON output.
    train_fn(seed) should return {"test_accuracy": float, ...}
    Saves incrementally after each seed to survive timeouts.
    """
    all_results = {}
    for seed in seeds:
        set_seed(seed)
        result = train_fn(seed=seed)
        all_results[f"seed_{seed}"] = result
        print(f"Seed {seed}: {result}")
        # Save after EACH seed (survive timeouts)
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)

    accs = [r["test_accuracy"] for r in all_results.values()]
    summary = {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "seeds": seeds,
        "accuracies": accs,
        "per_seed": all_results,
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"mean_accuracy: {summary['mean_accuracy']:.2f}")
    print(f"std_accuracy: {summary['std_accuracy']:.2f}")
    return summary
''',

    "statistical_comparison": '''
import numpy as np
from scipy import stats

def compare_methods(results_a, results_b, method_a_name, method_b_name):
    """Statistical comparison of two methods — paired t-test + Cohen's d.
    results_a, results_b: lists of accuracy values (one per seed).
    """
    a = np.array(results_a)
    b = np.array(results_b)
    t_stat, p_value = stats.ttest_rel(a, b)
    diff = a - b
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
    print(f"{method_a_name}: {np.mean(a):.2f} ± {np.std(a):.2f}")
    print(f"{method_b_name}: {np.mean(b):.2f} ± {np.std(b):.2f}")
    print(f"t_statistic: {t_stat:.4f}")
    print(f"p_value: {p_value:.4f}")
    print(f"cohens_d: {cohens_d:.4f}")
    sig = "significant" if p_value < 0.05 else "not significant"
    print(f"Result: {sig} (p={p_value:.4f})")
    return {"t_statistic": t_stat, "p_value": p_value, "cohens_d": cohens_d}

def compare_multiple_methods(all_results: dict, metric_key="test_accuracy"):
    """Compare ≥3 methods with Holm-Bonferroni correction.
    all_results: {"method_name": [acc_seed1, acc_seed2, ...], ...}
    Returns dict with pairwise comparisons and corrected p-values.
    """
    from itertools import combinations
    methods = list(all_results.keys())
    n = len(methods)
    if n < 2:
        return {"error": "Need ≥2 methods to compare"}

    # Pairwise comparisons
    comparisons = []
    p_values = []
    for a_name, b_name in combinations(methods, 2):
        a = np.array(all_results[a_name])
        b = np.array(all_results[b_name])
        t_stat, p_val = stats.ttest_rel(a, b)
        diff = a - b
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)
        comparisons.append({
            "method_a": a_name, "method_b": b_name,
            "t_statistic": float(t_stat), "p_value": float(p_val),
            "cohens_d": float(d),
        })
        p_values.append(p_val)

    # Holm-Bonferroni correction
    if len(p_values) > 1:
        sorted_indices = np.argsort(p_values)
        corrected = [0.0] * len(p_values)
        for rank, idx in enumerate(sorted_indices):
            corrected[idx] = min(1.0, p_values[idx] * (len(p_values) - rank))
        # Enforce monotonicity
        for i in range(1, len(sorted_indices)):
            idx = sorted_indices[i]
            prev_idx = sorted_indices[i-1]
            corrected[idx] = max(corrected[idx], corrected[prev_idx])
        for i, comp in enumerate(comparisons):
            comp["corrected_p"] = corrected[i]
            comp["holm_significant"] = corrected[i] < 0.05

    summary = {name: {"mean": float(np.mean(v)), "std": float(np.std(v))}
               for name, v in all_results.items()}
    best = max(summary, key=lambda k: summary[k]["mean"])

    result = {
        "n_methods": n, "n_comparisons": len(comparisons),
        "correction_method": "holm_bonferroni",
        "comparisons": comparisons, "summary": summary, "best_method": best,
    }
    print(f"Compared {n} methods ({len(comparisons)} pairs, Holm-Bonferroni corrected)")
    for comp in comparisons:
        sig = "SIG" if comp.get("holm_significant") else "ns"
        print(f"  {comp['method_a']} vs {comp['method_b']}: "
              f"p={comp['p_value']:.4f} (corrected={comp.get('corrected_p', 'N/A'):.4f}) "
              f"d={comp['cohens_d']:.3f} [{sig}]")
    return result
''',
}


def get_relevant_recipes(task_description: str) -> list[tuple[str, str]]:
    """Return relevant (name, code) recipe pairs for a task description."""
    keywords_map = {
        r"cifar": ["cifar10_dataloader", "training_loop"],
        r"mnist": ["mnist_dataloader", "training_loop"],
        r"sst.?2|sentiment|glue": ["sst2_dataloader"],
        r"vgg": ["vgg_cifar", "training_loop"],
        r"resnet": ["training_loop"],
        r"simple.*(cnn|conv|network)|cnn|conv": ["simple_cnn_cifar", "training_loop"],
        r"multi.?seed|[35]\s*seeds|multiple\s*seeds": ["multi_seed_eval"],
        r"t.?test|cohen|statistical|compare|versus|vs\.?": ["statistical_comparison"],
        r"train": ["training_loop"],
    }
    recipes = set()
    desc_lower = task_description.lower()
    for pattern, recipe_names in keywords_map.items():
        if re.search(pattern, desc_lower):
            recipes.update(recipe_names)

    # Always include multi_seed if seeds are mentioned
    if re.search(r"seed", desc_lower):
        recipes.add("multi_seed_eval")

    # If neural/deep learning task, at least give training loop
    if not recipes and any(k in desc_lower for k in ["neural", "model", "deep",
                                                       "cnn", "mlp", "transformer"]):
        recipes.add("training_loop")

    return [(name, RECIPES[name]) for name in recipes if name in RECIPES]


def format_recipes_for_prompt(task_description: str) -> str:
    """Format relevant recipes as a prompt section."""
    relevant = get_relevant_recipes(task_description)
    if not relevant:
        return ""

    parts = ["## Verified Code Recipes (USE THESE — they are pre-tested and working)"]
    for name, code in relevant:
        parts.append(f"\n### Recipe: {name}\n```python{code}```")
    parts.append("\n**IMPORTANT: Use these recipes as building blocks. "
                 "Do NOT rewrite data loading or training loops from scratch.**")
    return "\n".join(parts)
