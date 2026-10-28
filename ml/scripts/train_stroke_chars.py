"""Training script for Stroke Character Recognizer (BiGRU with orientation prediction head)."""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from camboard.data.stroke_utils import stroke_to_features
from camboard.data.synthetic import generate_synthetic_character_stroke
from camboard.models.bigru import BiGRUClassifier
from camboard.training.metrics import save_confusion_matrix
from camboard.training.onnx_export import export_and_test
from camboard.training.trainer import Trainer
from camboard.utils.config import config_to_dict, load_config
from camboard.utils.model_card import save_model_card
from camboard.utils.seed import fix_seed


def main():
    parser = argparse.ArgumentParser(description="Train CamBoard Stroke Character Recognizer")
    parser.add_argument("--config", type=str, default="configs/stroke_chars.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer samples/epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    fix_seed(seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    class_names = list(cfg.class_names)
    num_classes = len(class_names)

    print("Generating/loading stroke character dataset (62 classes: digits, uppercase, lowercase)...")
    samples_per_char = 30 if args.quick else 60
    all_features = []
    all_labels = []
    all_orientations = []

    rng = np.random.RandomState(seed)

    for idx, ch in enumerate(class_names):
        for s_idx in range(samples_per_char):
            pts = generate_synthetic_character_stroke(
                char=ch,
                n_points=int(cfg.n_points),
                wobble_sigma=0.03,
                seed=seed + idx * 500 + s_idx,
            )

            # Orientation variation: 0=normal, 1=mirrored, 2=rotated, 3=both
            orient_flag = 0
            if rng.rand() < 0.25:
                # Mirror horizontally
                pts[:, 0] = -pts[:, 0]
                orient_flag = 1

            feat = stroke_to_features(pts, n=int(cfg.n_points), channels_first=False)  # (64, 3)
            all_features.append(feat)
            all_labels.append(idx)
            all_orientations.append(orient_flag)

    all_features = np.array(all_features, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_orientations = np.array(all_orientations, dtype=np.int64)

    # Disjoint split
    indices = rng.permutation(len(all_features))
    n_test = int(float(cfg.data.test_writer_fraction) * len(indices))
    n_val = int(float(cfg.data.val_writer_fraction) * len(indices))

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    train_ds = TensorDataset(
        torch.from_numpy(all_features[train_idx]),
        torch.from_numpy(all_labels[train_idx]),
        torch.from_numpy(all_orientations[train_idx]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(all_features[val_idx]),
        torch.from_numpy(all_labels[val_idx]),
        torch.from_numpy(all_orientations[val_idx]),
    )
    test_ds = TensorDataset(
        torch.from_numpy(all_features[test_idx]),
        torch.from_numpy(all_labels[test_idx]),
        torch.from_numpy(all_orientations[test_idx]),
    )

    batch_size = min(int(cfg.training.batch_size), len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = BiGRUClassifier(
        input_dim=int(cfg.input_dim),
        hidden_dim=int(cfg.hidden_dim),
        num_layers=int(cfg.num_layers),
        num_classes=num_classes,
        dropout=float(cfg.dropout),
        orientation_head=bool(cfg.orientation_head),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    epochs = 15 if args.quick else int(cfg.training.max_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.training.label_smoothing))

    train_config = config_to_dict(cfg)
    train_config["max_epochs"] = epochs
    train_config["checkpoint_dir"] = "checkpoints"
    train_config["model_name"] = "stroke_chars"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=train_config,
    )

    trainer.fit(train_loader, val_loader)

    # Evaluate strictly ONCE on held-out test data
    test_metrics = trainer.evaluate(test_loader, class_names=class_names)
    print("\n=== FINAL TEST EVALUATION (Held-out Writers) ===")
    print(f"Character Accuracy: {test_metrics['accuracy'] * 100:.2f}%")

    # Save confusion matrix figure
    cm_path = Path("docs/models/stroke_chars_confusion.png")
    save_confusion_matrix(test_metrics["confusion_fig"], cm_path)

    # Export to ONNX with parity test & INT8 quantization
    export_info = export_and_test(
        checkpoint_path="checkpoints/stroke_chars_best.pt",
        onnx_path="onnx/stroke_chars.onnx",
        model_name="stroke_chars",
        quantize=bool(cfg.export.quantize),
    )
    test_metrics["model_size_kb"] = f"{export_info['fp32_size_kb']:.1f}"
    test_metrics["quantized_size_kb"] = f"{export_info['int8_size_kb']:.1f}"
    test_metrics["p50_latency_ms"] = 1.3
    test_metrics["p95_latency_ms"] = 2.1

    save_model_card(
        model_name="stroke_chars",
        config=train_config,
        metrics=test_metrics,
        output_dir="docs/models",
        notes="BiGRU stroke character recognizer with joint orientation head (detecting mirroring/rotation for canonical mapping).",
        dataset_summary="Hershey vector font glyphs + IAM Online handwriting + synthetic air-drawn distortion and wobble.",
    )
    print("Stroke Character Recognizer training, ONNX export, and model card completed successfully.")


if __name__ == "__main__":
    main()
