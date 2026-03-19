"""训练日志记录，便于观察训练曲线"""
import json
from pathlib import Path


class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

    def log(self, step: int, **kwargs):
        record = {"step": step, **kwargs}
        self.history.append(record)
        return record

    def save(self, filename: str = "history.json"):
        path = self.log_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
