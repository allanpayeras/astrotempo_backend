import os

from typing import List
from fastapi import FastAPI, HTTPException

TILE_ROOT = "/var/www/tiles"
VALID_PRODUCTS = {"cod", "acha", "acht"}

app = FastAPI()


def get_available_timestamps(product: str) -> List[str]:
    product_path = os.path.join(TILE_ROOT, product)

    if not os.path.isdir(product_path):
        return []

    # Only keep directories (timestamps)
    timestamps = [
        name
        for name in os.listdir(product_path)
        if os.path.isdir(os.path.join(product_path, name))
    ]

    return sorted(timestamps)


@app.get("/timestamps/{product}")
def list_timestamps(product: str):
    if product not in VALID_PRODUCTS:
        raise HTTPException(status_code=400, detail="Invalid product")

    timestamps = get_available_timestamps(product)

    if not timestamps:
        raise HTTPException(status_code=404, detail="No timestamps found")

    return {
        "product": product,
        "count": len(timestamps),
        "timestamps": timestamps,
    }
