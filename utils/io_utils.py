
"""
io_utils.py
Basic helpers for I/O and array conversions.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import json
import csv

def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def write_csv(rows: List[Dict[str, Any]], path: str, fieldnames: Optional[List[str]] = None) -> None:
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
