"""Train a YOLO11 segmentation model for concrete crack detection."""

from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11l-seg.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs" / "data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Examples: 0, cpu, 0,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="yolo11l_concrete_cracks")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        task="segment",
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs" / "segment"),
        name=args.name,
        pretrained=True,
        plots=True,
        seed=0,
    )


if __name__ == "__main__":
    main()

