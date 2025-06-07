"""
Sentinel-2 → RGB  (+ optional Cropland Data Layer & binary mask)  •  Fused UDF
───────────────────────────────────────────────────────────────────────────────
• Adaptive native / decimated resolution   (10 m @ Z13, coarser when zoomed out)
• Temporal aggregate   :  percentile | mean | median | min
• Indices supported    :  NDVI · BAI · NDSI · NDWI · NDMI · NDBI · MSAVI
      – Each index can be *individually boosted* via the `idx_boost` dict.

╭─ Cropland options ───────────────────────────────────────────────────────────╮
│ cdl_alpha=True,  cdl_colored=False  → 4-th band = raw CDL class IDs          │
│ cdl_alpha=True,  cdl_colored=True   → Sentinel RGB under CDL RGBA overlay    │
│ cdl_as_mask=True                    → 4-th band = 0/255 mask for crops in    │
│                                        `crop_type=` (RGB remains untouched)  │
╰──────────────────────────────────────────────────────────────────────────────╯

Example crop strings:  "corn", "soybean", "alfalfa", "water",  "corn,alfalfa"

Gabriel Durkin  |  31 May 2025
"""

# ───── global switches ─────
USE_LOCAL_PCTL = False        # blend global P99 with per-tile P98?
LOCAL_CLIP_PCT = 98
# ───────────────────────────

import fused, numpy as np, cv2, planetary_computer, pystac_client, re
from odc.stac import load
from pystac.extensions.eo import EOExtension as eo
from gabeutils import adjust_gamma, increment_month_with_year, nan_percentile, S2_stats

# ─── CDL helpers (cached) ─────────────────────────────────────────────────────
common = fused.load(
    "https://github.com/fusedio/udfs/tree/b3a7ff8/public/common/"
).utils


@fused.cache
def _get_cdl_meta():
    import pandas as pd, requests, json
    url = "https://storage.googleapis.com/earthengine-stac/catalog/USDA/USDA_NASS_CDL.json"
    band = requests.get(url).json()["summaries"]["eo:bands"][0]
    return pd.DataFrame(band["gee:classes"]).set_index("value")          # description • color


@fused.cache
def _tokens_to_vals(token_csv: str):
    """Comma list of crop keywords → list of CDL class IDs (empty ⇒ all)."""
    if not token_csv:
        return []
    df, vals = _get_cdl_meta(), set()
    for tok in (t.strip() for t in token_csv.split(",") if t.strip()):
        pat = re.compile(rf"\b{re.escape(tok)}s?\b", re.IGNORECASE)       # tolerate plurals
        vals.update(df[df.description.str.contains(pat)].index.tolist())
    if not vals:
        print(f"[CDL] No match for “{token_csv}” – showing all crops.")
    return sorted(vals)


def _cdl_read(bounds, year, out_shape, token_csv, colored=True):
    """
    Read the Cropland Data Layer for *bounds* at *out_shape*.

    Returns
    -------
    (H,W)   uint8   … raw class IDs   (colored=False)
    (4,H,W) uint8   … RGBA image      (colored=True)
    """
    h, w = out_shape
    if h * w == 0:
        return (np.zeros((4, 0, 0), np.uint8)
                if colored else np.zeros((0, 0), np.uint8))

    tile = common.get_tiles(bounds, clip=True)
    path = f"s3://fused-asset/data/cdls/{year}_30m_cdls.tif"
    data = common.read_tiff(tile, path, output_shape=(h, w),
                            return_colormap=True, cache_max_age="90d")
    if not data:                       # tile outside CONUS → transparent/zeros
        return (np.zeros((4, h, w), np.uint8)
                if colored else np.zeros((h, w), np.uint8))

    arr_int, meta = data
    keep = _tokens_to_vals(token_csv)
    if keep:
        arr_int[~np.isin(arr_int, keep)] = 0

    # — legend -----------------------------------------------------------------
    df, uniq = _get_cdl_meta(), np.unique(arr_int[arr_int > 0])
    if uniq.size:
        print("[CDL legend]")
        for v in uniq:
            print(f"{v:>4}  {df.loc[v,'description']:<35}  #{df.loc[v,'color']}")
    else:
        print("[CDL] No matching pixels in this tile.")

    if not colored:
        return arr_int.astype(np.uint8)

    cmap = meta["colormap"]                               # {int: (R,G,B,A)}
    rgba = np.array([cmap[v] for v in arr_int.flat], np.uint8)\
              .reshape(arr_int.shape + (4,)).transpose(2, 0, 1)
    return rgba


# helper: pick Sentinel-2 res for this WM tile ---------------------------------
def _resolution_from_bounds(bounds, tile_px=256):
    xmin, ymin, xmax, ymax = bounds
    res_m = (xmax - xmin) / tile_px
    zoom  = round(np.log2(156543.03392804097 / res_m))
    return int(max(10, 10 * 2 ** (13 - zoom)))            # 10 m native ≥ Z13


# ───────────────────────────────  MAIN  UDF  ──────────────────────────────────
@fused.cache
def udf(
    bounds: fused.types.Bounds,
    # RGB / index recipe -------------------------------------------------------
    pR="built_up", pG="water", pB="msavi", pA=None,
    idx_boost={"built_up":1.9, "water":1.15, "msavi":0.9},
    msavi_min=-0.1, msavi_max=0.6,
    gamma=0.35, brightness=1.25, saturation_factor=1.3,
    nd_min=-0.65, nd_max=0.3,
    # temporal -----------------------------------------------------------------
    year=2024, month="06", month_window=0,
    time_slice_count=7, ordering="cloud_ordered",
    agg_type="percentile", percentile=20, qa=False,
    # Cropland switches --------------------------------------------------------
    cdl_alpha=True,          # add CDL band / overlay?
    cdl_year=2023,
    cdl_colored=False,        # True: RGBA overlay   False: raw IDs band
    cdl_as_mask=True,        # True: alpha-mask RGB with selected crops
    crop_type="corn",
):
    # ───── bookkeeping
    s2_bands = ["AOT","B01","B02","B03","B04","B05","B06","B07",
                "B08","B09","B11","B12","B8A","SCL","WVP","visual"]
    index_bands = {
        "veg":["B08","B04"], "burn":["B08","B12"], "burn_r":["B12","B08"],
        "snow":["B03","B11"], "glacier":["B03","B04"], "water":["B03","B08"],
        "moisture":["B08","B11"], "built_up":["B12","B11","B03"],
        "new_built":["B04","B12","B08"], "msavi":["B08","B04"],
    }
    cloud_codes = [3, 8, 9, 10]

    # ───── date window
    m2, y2 = increment_month_with_year(month, year, month_window)
    time_str = f"{year}-{month}-01/{y2}-{m2}-28"

    # ───── decide what to load
    chans, pure, to_load = [], [], set()
    for ch in [pR, pG, pB] + ([pA] if pA else []):
        chans.append(ch)
        if ch in index_bands:
            to_load.update(index_bands[ch])
        elif ch in s2_bands:
            pure.append(ch); to_load.add(ch)
        else:
            raise ValueError(f"Unknown band/index «{ch}»")
    if qa:
        to_load.add("SCL")

    # ───── STAC query (single page → fast)
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace)
    items = list(cat.search(
        collections=["sentinel-2-l2a"],
        bbox=bounds, datetime=time_str, limit=time_slice_count).get_items())
    if not items:
        raise RuntimeError("No Sentinel-2 scenes intersect this tile.")

    key = (lambda i: eo.ext(i).cloud_cover) if ordering=="cloud_ordered" \
          else (lambda i: i.datetime)
    items.sort(key=key, reverse=(ordering=="time_ordered"))

    # ───── load cube
    res = _resolution_from_bounds(bounds)
    ds  = load(items, bands=list(to_load),
               crs="EPSG:3857", resolution=res, bbox=bounds, chunks={})
    if qa and "SCL" in ds:
        ds = ds.where(~ds["SCL"].isin(cloud_codes))
    ds = ds.where(ds != 0, np.nan)

    # ───── temporal aggregate
    if agg_type == "percentile":
        agg = {b: nan_percentile(ds[b].values, percentile)[0]
               for b in ds.data_vars}
        ds  = {k: v/1e4 for k, v in agg.items()}
    elif agg_type in ("mean", "median", "min"):
        tmp = getattr(ds, agg_type)("time", skipna=True)
        ds  = {b: tmp[b].values/1e4 for b in tmp.data_vars}
    else:
        raise ValueError("agg_type must be percentile | mean | median | min")

    # ───── stretch helper
    def _stretch(arr0, band, eps=1e-4):
        arr = arr0.astype(np.float32);  arr[np.isnan(arr)] = 0
        hi_g = S2_stats[band]["p99"]/1e4
        hi_l = np.nanpercentile(arr, LOCAL_CLIP_PCT) if USE_LOCAL_PCTL else hi_g
        return np.clip(arr/max(eps, min(hi_l, hi_g)), 0, 1)*255

    # ───── build channels (RGB / indices)
    layers = []
    for ch in chans:
        if ch in pure:
            layers.append(_stretch(ds[ch], ch))
        else:
            need = {b: _stretch(ds[b], b)/255 for b in index_bands[ch]}

            if ch in ("veg","burn","burn_r","snow","water","moisture","glacier"):
                n, d = need[index_bands[ch][0]], need[index_bands[ch][1]]
                idx  = (n - d) / (n + d + 1e-4)
                layers.append(np.clip((idx - nd_min)/(nd_max - nd_min), 0, 1)
                              * idx_boost.get(ch, 1.0) * 255)

            elif ch == "msavi":
                nir, red = need["B08"], need["B04"]
                t  = 2*nir + 1
                ms = (t - np.sqrt(np.maximum(t*t - 8*(nir-red), 0))) / 2
                sc = np.clip((ms - msavi_min)/(msavi_max - msavi_min), 0, 1)
                layers.append(sc * idx_boost.get("msavi", 1.0) * 255)

            elif ch == "built_up":
                B12, B11, B03 = need["B12"], need["B11"], need["B03"]
                idx = (B12 - (B11/(B03+1e-4))) / \
                      (B12 + (B11/(B03+1e-4)) + 1e-4)
                layers.append(np.clip((idx - nd_min)/(nd_max - nd_min), 0, 1)
                              * idx_boost.get("built_up", 1.0) * 255)

            elif ch == "new_built":
                B04, B12, B08 = need["B04"], need["B12"], need["B08"]
                idx = (B04*B12)/(B08 + 1e-4)
                layers.append(np.clip(idx/0.85, 0, 1)
                              * idx_boost.get("new_built", 1.0) * 255)

    # ───── Cropland: IDs / overlay / mask
    if cdl_alpha or cdl_as_mask:
        h, w   = layers[0].shape
        cdl_ids = _cdl_read(bounds, cdl_year, (h, w),
                            token_csv=crop_type, colored=False)

    # —— binary alpha mask (does NOT modify RGB) ————————————————
    if cdl_as_mask:
        alpha_mask = (cdl_ids > 0).astype(np.uint8) * 255  # 0 / 255
        if len(layers) == 3:               # add new alpha band
            layers.append(alpha_mask)
        else:                              # overwrite existing α
            layers[3] = alpha_mask

    # —— CDL overlay or raw IDs band ————————————————————————————
    elif cdl_alpha:        # only if *not* already handled by mask
        if cdl_colored:
            cdl_rgba = _cdl_read(bounds, cdl_year, (h, w),
                                 token_csv=crop_type, colored=True)
            rgba_s2  = np.stack(layers, -1).astype(np.uint8)
            if rgba_s2.shape[-1] == 3:      # add opaque α
                rgba_s2 = np.concatenate(
                    [rgba_s2, np.full((h, w, 1), 255, np.uint8)], -1)
            alpha = cdl_rgba[3] > 0
            rgba_s2[alpha] = cdl_rgba[:, alpha].T
            layers = [rgba_s2[..., i] for i in range(4)]
        else:
            layers.append(cdl_ids.astype(np.uint8))

    # ───── cosmetic tweaks
    rgb = np.stack(layers[:3], -1)
    rgb = adjust_gamma(np.clip(brightness*rgb, 0, 255).astype(np.uint8),
                       gamma=gamma)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1]*saturation_factor, 0, 255)
    rgb_final = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    for i in range(3):
        layers[i][:] = rgb_final[..., i]

    out = np.stack(layers, -1).astype(np.uint8)   # (H,W,C)
    return out.transpose(2, 0, 1), bounds
