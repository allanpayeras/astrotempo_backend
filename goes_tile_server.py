from fastapi import FastAPI, HTTPException, Response
from rio_tiler.io import COGReader
from rio_tiler.utils import render
from pathlib import Path

app = FastAPI()

COG_DIR = Path("./goes_cog")


@app.get("/tiles/{timestamp}/{z}/{x}/{y}.png")
def tile(timestamp: str, z: int, x: int, y: int):
    cog_path = COG_DIR / f"goes_{timestamp}_cog.tif"
    if not cog_path.exists():
        raise HTTPException(status_code=404, detail="COG not found")

    with COGReader(cog_path) as cog:
        img = cog.tile(x, y, z, tilesize=256)

    data = render(img.data, mask=img.mask, format="PNG")

    return Response(content=data, media_type="image/png")
