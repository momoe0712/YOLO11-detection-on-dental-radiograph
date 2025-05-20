from ultralytics import YOLO
import os
from ultralytics.utils.plotting import plot_results
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class EarlyStopException(Exception):
    pass

def train_model():
    model = YOLO("--YOUR PRE-TRAINED MODEL PATH--") #example: pre-trained/yolo11x.pt

    # Custom early stopping callback
    class EarlyStopOnMap:
        def __init__(self, threshold=0.70):
            self.threshold = threshold

        def __call__(self, trainer):
            metrics = trainer.metrics
            if metrics and 'metrics/mAP50-95(B)' in metrics:
                current_map = metrics['metrics/mAP50-95(B)']
                print(f"[Callback] Current mAP50-95: {current_map:.4f}")
                if current_map >= self.threshold:
                    print(f"[Callback] mAP50-95 reached {current_map:.4f} (threshold: {self.threshold}) — stopping early.")
                    raise EarlyStopException

    model.add_callback('on_train_epoch_end', EarlyStopOnMap(threshold=0.70))

    try:
        results = model.train(
            data="--YOUR DATASET PATH--", #example: dataset/data.yaml
            epochs=500,
            imgsz=640,
            device=0,
            workers=1
        )
    except EarlyStopException:
        print("[Info] Early stopping triggered successfully — training halted.")

        # Run validation
        print("[Info] Running final validation to generate visual outputs...")
        model.val(data="dataset/data.yaml", imgsz=640, device=0)

        # Generate results.png manually
        log_dir = Path(model.trainer.save_dir)
        print(f"[Info] Generating results.png in {log_dir}")
        plot_results(log_dir / 'results.csv')  # FIXED

if __name__ == "__main__":
    train_model()
