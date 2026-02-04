from fastapi import FastAPI, HTTPException, Response
from rio_tiler.io import COGReader
from rio_tiler.utils import render
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

COG_DIR = Path("./goes_cog")
# Allowed satellite products
VALID_PRODUCTS = {"cod", "acha", "acht"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/tiles/{product}/{timestamp}/{z}/{x}/{y}.png")
def tile(
    product: str,
    timestamp: str,
    z: int,
    x: int,
    y: int,
):
    if product not in VALID_PRODUCTS:
        raise HTTPException(status_code=400, detail="Invalid product")

    cog_path = COG_DIR / f"goes_{timestamp}_{product}_cog.tif"
    if not cog_path.exists():
        raise HTTPException(status_code=404, detail="COG not found")

    with COGReader(cog_path) as cog:
        img = cog.tile(x, y, z, tilesize=256)

    data = render(img.data, format="PNG")
    return Response(
        content=data,
        media_type="image/png",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Cache-Control": "public, max-age=86400",
        },
    )
