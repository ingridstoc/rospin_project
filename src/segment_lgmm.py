import os
from pathlib import Path
import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
except Exception as e:
    raise SystemExit("❌ Instalează scikit-learn: pip install scikit-learn\n" + str(e))

IMG_EXTS = [".tiff", ".tif", ".png", ".jpg", ".jpeg"]

def load_meta(meta_path: Path)->pd.DataFrame:
    p=Path(meta_path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower()==".csv":
        df=pd.read_csv(p)
    elif p.suffix.lower() in (".xls", ".xlsx"):
        df=pd.read_excel(p)
    else:
        raise ValueError("Meta acceptă .csv/.xls/.xlsx")
    cols={c.lower(): c for c in df.columns}
    if "id" not in cols or "path" not in cols:
        raise KeyError("Meta trebuie coloanele 'id' și 'path'")
    return df

def find_image(root_dir: Path, subpath_str: str, base_id: str)->Path | None:
    sub=Path(str(subpath_str).strip().replace("\\", os.sep).replace("/", os.sep))
    for ext in IMG_EXTS:
        p=root_dir / sub / f"{base_id}{ext}"
        if p.exists():
            return p
    return None

def ensure_positive(img: np.ndarray)->np.ndarray:
    x=img.astype(np.float32)
    if np.any(x<=0):
        pos=x[x>0]
        x[x<=0]=np.min(pos) if pos.size else 1.0
    return x

def to_log(gray: np.ndarray)->np.ndarray:
    return np.log1p(ensure_positive(gray))

def read_gray(p: Path)->np.ndarray:
    im=cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(p)
    return im

def gaussian_if(x: np.ndarray, sigma: float)->np.ndarray:
    if sigma and sigma>0:
        return cv2.GaussianBlur(x, (0, 0), sigmaX=float(sigma))
    return x

def nodata_from_border_zeros(img_u8: np.ndarray) -> np.ndarray:
    z=(img_u8==0).astype(np.uint8)
    H, W=z.shape
    vis=np.zeros_like(z, dtype=np.uint8)
    q=deque()

    for i in range(H):
        if z[i, 0]: q.append((i, 0))
        if z[i, W-1]: q.append((i, W-1))
    for j in range(W):
        if z[0, j]: q.append((0, j))
        if z[H-1, j]: q.append((H-1, j))

    while q:
        i, j=q.popleft()
        if vis[i, j]: continue
        vis[i, j]=1
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ni, nj=i+di, j+dj
            if 0<=ni<H and 0<=nj<W and z[ni, nj] and not vis[ni, nj]:
                q.append((ni, nj))
    return vis.astype(bool)

# ---------- Otsu în log (pe pixeli valizi) ----------

def otsu_mask_from_log(x_log: np.ndarray, valid: np.ndarray | None=None)->np.ndarray:
    if valid is None:
        valid=np.isfinite(x_log)
    x=x_log[valid]
    if x.size==0:
        return np.zeros_like(x_log, dtype=np.uint8)
    lo, hi=np.percentile(x, [0.5, 99.5])
    if hi<=lo:
        lo, hi=float(x.min()), float(x.max() + 1e-6)
    x8=np.clip((x_log-lo)/(hi-lo+1e-12), 0, 1)
    x8=(x8*255).astype(np.uint8)
    th, _=cv2.threshold(x8[valid], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sea=(x8<th).astype(np.uint8)      # 1=SEA
    sea[~valid]=0                       # NoData => LAND
    return sea


def local_variance_log(x_log: np.ndarray, win: int=7)->np.ndarray:
    k=(win, win)
    m=cv2.boxFilter(x_log, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    m2=cv2.boxFilter(x_log*x_log, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    v=m2-m*m
    v[v<0]=0
    return v


def mask_kmeans_log(x_log: np.ndarray, max_samples: int=100_000, n_init: int=1, seed: int=0)->np.ndarray:
    X=x_log.reshape(-1, 1)
    if X.shape[0]>max_samples:
        idx=np.random.RandomState(seed).choice(X.shape[0], max_samples, replace=False)
        X_fit=X[idx]
    else:
        X_fit=X
    km=KMeans(n_clusters=2, n_init=n_init, random_state=seed, algorithm="elkan")
    km.fit(X_fit)
    labels=km.predict(X).reshape(x_log.shape)
    means=[x_log[labels==i].mean() if np.any(labels==i) else np.inf for i in (0, 1)]
    sea_label=int(np.argmin(means))
    return (labels==sea_label).astype(np.uint8)

def mask_varlog_otsu(x_log: np.ndarray, win: int=7)->np.ndarray:
    c2=local_variance_log(x_log, win)
    x=c2[np.isfinite(c2)]
    if x.size==0:
        return np.zeros_like(c2, dtype=np.uint8)
    lo, hi=np.percentile(x, [0.5, 99.5])
    if hi<=lo:
        lo, hi=float(x.min()), float(x.max()+1e-6)
    x8=np.clip((c2-lo)/(hi-lo+1e-12), 0, 1)
    x8=(x8*255).astype(np.uint8)
    _, th=cv2.threshold(x8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return (x8<th).astype(np.uint8)

def majority_vote(masks: list[np.ndarray]) -> np.ndarray:
    stack=np.stack(masks, axis=0).astype(np.uint8)
    return (stack.sum(axis=0)>=2).astype(np.uint8)


def remove_small_components(bin_img: np.ndarray, min_area: int)->np.ndarray:
    if min_area<=0: 
        return bin_img.astype(np.uint8)
    n, lab, stats, _=cv2.connectedComponentsWithStats(bin_img.astype(np.uint8), connectivity=8)
    out=np.zeros_like(bin_img, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA]>=min_area:
            out[lab==i]=1
    return out

def fill_small_components(bin_img: np.ndarray, max_area: int, fill_value: int)->np.ndarray:
    if max_area<=0:
        return bin_img
    n, lab, stats, _=cv2.connectedComponentsWithStats(bin_img.astype(np.uint8), connectivity=8)
    out=bin_img.copy().astype(np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max_area:
            out[lab==i]=fill_value
    return out

def postprocess(mask: np.ndarray,
                close_k: int=7,
                min_area_sea: int=300,
                min_area_land: int=0,
                hole_sea: int=1200,
                hole_land: int=300)->np.ndarray:
    m=mask.astype(np.uint8)

    if close_k and close_k>1:
        k=cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        m=cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    m=remove_small_components(m, min_area_sea)

    inv=remove_small_components(1-m, min_area_land)
    m=1-inv

    inv=fill_small_components(1 - m, hole_sea, fill_value=0)  
    m=1-inv

    m=fill_small_components(m, hole_land, fill_value=0)

    return m.astype(np.uint8)


def segment_with_gmm_sklearn(
    image_gray: np.ndarray,
    close_k: int=7,
    min_area_sea: int=300,
    min_area_land: int=0,
    hole_sea: int=1200,
    hole_land: int=300,
    trim_top_pct: float=1.0,
    cov_type: str="diag",
    random_state: int=0,
    n_init: int=1,
    bic_margin: float=0.0,
    all_sea_thresh: float=0.90,
    all_land_thresh: float=0.05,
    fit_max: int=50_000,
    max_iter: int=50,
    tol: float=1e-3,
    reg_covar: float=1e-6,
    pre_gauss: float=0.0,
)->np.ndarray:

    nodata=nodata_from_border_zeros(image_gray)     
    x_log=gaussian_if(to_log(image_gray), pre_gauss)
    valid=(~nodata)&np.isfinite(x_log)

    vals=x_log[valid]
    if vals.size==0:
        return np.zeros_like(image_gray, dtype=np.uint8)

    if 0<trim_top_pct<100:
        cut=np.percentile(vals, 100-float(trim_top_pct))
        fit_x=vals[vals<=cut]
        if fit_x.size<50:
            fit_x=vals
    else:
        fit_x=vals

    if fit_x.size>fit_max:
        rng=np.random.default_rng(random_state)
        pick=rng.choice(fit_x.size, size=fit_max, replace=False)
        fit_x=fit_x[pick]

    Xfit=fit_x.reshape(-1, 1)

    p30, p70=np.percentile(fit_x, [30, 70])
    means2=np.array([[p30], [p70]], dtype=np.float64)
    mean1 =np.array([[fit_x.mean()]], dtype=np.float64)

    gm1=GaussianMixture(n_components=1, covariance_type=cov_type, random_state=random_state,
                          n_init=n_init, max_iter=max_iter, tol=tol, reg_covar=reg_covar,
                          means_init=mean1)
    gm1.fit(Xfit); bic1=gm1.bic(Xfit)

    gm2=GaussianMixture(n_components=2, covariance_type=cov_type, random_state=random_state,
                          n_init=n_init, max_iter=max_iter, tol=tol, reg_covar=reg_covar,
                          means_init=means2)
    gm2.fit(Xfit); bic2=gm2.bic(Xfit)

    if bic1<=bic2+bic_margin:
        tmp=otsu_mask_from_log(x_log, valid=valid)
        sea_frac=float(tmp[valid].mean())
        if sea_frac>=all_sea_thresh:
            mask=np.zeros_like(image_gray, dtype=np.uint8); mask[valid]=1
        elif sea_frac<=all_land_thresh:
            mask=np.zeros_like(image_gray, dtype=np.uint8)
        else:
            mask=tmp.astype(np.uint8)
    else:
        Xvalid=vals.reshape(-1, 1)
        proba=gm2.predict_proba(Xvalid)
        means=gm2.means_.reshape(-1)
        sea_comp=int(np.argmin(means))
        sea_valid=(proba[:, sea_comp] >= proba[:, 1 - sea_comp]).astype(np.uint8)
        mask=np.zeros_like(image_gray, dtype=np.uint8)
        mask[valid]=sea_valid

    mask[nodata]=0 # NoData = LAND

    mask=postprocess(mask,
                       close_k=close_k,
                       min_area_sea=min_area_sea,
                       min_area_land=min_area_land,
                       hole_sea=hole_sea,
                       hole_land=hole_land)
    return mask


def _seg_lgmm_one(root_dir, out_root, fid, sub, kwargs):
    try:
        src=find_image(root_dir, sub, fid)
        if src is None:
            return "miss"
        img=read_gray(src)
        mask=segment_with_gmm_sklearn(img, **kwargs)
        dst=out_root/Path(sub)/f"{Path(src).stem}_mask.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), (mask * 255).astype(np.uint8))
        return "ok"
    except Exception:
        return "fail"

def _seg_vote_one(root_dir, out_root, fid, sub, win, close_k, min_area_sea, min_area_land, kmeans_samples):
    try:
        src=find_image(root_dir, sub, fid)
        if src is None:
            return "miss"
        img=read_gray(src)
        x_log=to_log(img)
        m1=mask_kmeans_log(x_log, max_samples=kmeans_samples)
        m2=otsu_mask_from_log(x_log, valid=np.isfinite(x_log))
        m3=mask_varlog_otsu(x_log, win=win)
        voted=majority_vote([m1, m2, m3])
        mask=postprocess(voted, close_k=close_k,
                           min_area_sea=min_area_sea, min_area_land=min_area_land,
                           hole_sea=1200, hole_land=300)
        dst=out_root/Path(sub)/f"{Path(src).stem}_vote_mask.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), (mask * 255).astype(np.uint8))
        return "ok"
    except Exception:
        return "fail"


def run_seg_lgmm(root_dir: Path, meta_path: Path, out_root: Path,
                 close_k=7, min_area_sea=300, min_area_land=0,
                 hole_sea=1200, hole_land=300,
                 trim_top=1.0, cov="diag", seed=0, n_init=1,
                 bic_margin=-5.0, all_sea=0.90, all_land=0.05,
                 fit_max=50_000, max_iter=50, tol=1e-3, reg_covar=1e-6,
                 pre_gauss=0.0,
                 workers=1, dry_run=False):
    df=load_meta(meta_path)
    root_dir, out_root=Path(root_dir), Path(out_root)
    rows=[(str(r._asdict()[{k.lower(): k for k in r._asdict().keys()}["id"]]).strip(),
             str(r._asdict()[{k.lower(): k for k in r._asdict().keys()}["path"]]).strip())
            for r in df.itertuples(index=False)]
    if dry_run:
        print(f"[dry-run] ar procesa {len(rows)} imagini."); return

    kwargs=dict(close_k=close_k, min_area_sea=min_area_sea, min_area_land=min_area_land,
                  hole_sea=hole_sea, hole_land=hole_land,
                  trim_top_pct=trim_top, cov_type=cov, random_state=seed, n_init=n_init,
                  bic_margin=bic_margin, all_sea_thresh=all_sea, all_land_thresh=all_land,
                  fit_max=fit_max, max_iter=max_iter, tol=tol, reg_covar=reg_covar,
                  pre_gauss=pre_gauss)

    ok=miss=fail=0
    if workers and workers>1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut=[ex.submit(_seg_lgmm_one, root_dir, out_root, fid, sub, kwargs) for fid, sub in rows]
            for f in tqdm(as_completed(fut), total=len(fut), desc="seg-lgmm (paralel)"):
                res=f.result(); ok+=(res=="ok"); miss+=(res=="miss"); fail+=(res=="fail")
    else:
        for fid, sub in tqdm(rows, total=len(rows), desc="seg-lgmm"):
            res=_seg_lgmm_one(root_dir, out_root, fid, sub, kwargs)
            ok+=(res=="ok"); miss+=(res=="miss"); fail+=(res=="fail")
    print(f"\nLGMM done. saved={ok}  missed={miss}  failed={fail}")

def run_seg_vote(root_dir: Path, meta_path: Path, out_root: Path,
                 win=7, close_k=7, min_area_sea=300, min_area_land=0,
                 kmeans_samples=100_000, workers=1, dry_run=False):
    df=load_meta(meta_path)
    root_dir, out_root=Path(root_dir), Path(out_root)
    rows=[(str(r._asdict()[{k.lower(): k for k in r._asdict().keys()}["id"]]).strip(),
             str(r._asdict()[{k.lower(): k for k in r._asdict().keys()}["path"]]).strip())
            for r in df.itertuples(index=False)]
    if dry_run:
        print(f"[dry-run] ar procesa {len(rows)} imagini."); return

    ok=miss=fail=0
    if workers and workers>1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut=[ex.submit(_seg_vote_one, root_dir, out_root, fid, sub, win, close_k, min_area_sea, min_area_land, kmeans_samples)
                   for fid, sub in rows]
            for f in tqdm(as_completed(fut), total=len(fut), desc="seg-vote (paralel)"):
                res=f.result(); ok+=(res=="ok"); miss+=(res=="miss"); fail+=(res=="fail")
    else:
        for fid, sub in tqdm(rows, total=len(rows), desc="seg-vote"):
            res=_seg_vote_one(root_dir, out_root, fid, sub, win, close_k, min_area_sea, min_area_land, kmeans_samples)
            ok+=(res=="ok"); miss+=(res=="miss"); fail+=(res=="fail")
    print(f"\nVoting done. saved={ok}  missed={miss}  failed={fail}")


def overlay_bgr(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.45, color_bgr=(255, 0, 0)) -> np.ndarray:
    base=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape)==2 else gray.copy()
    m=(mask>0).astype(np.uint8)
    color=np.zeros_like(base); color[:, :, 0] = m*color_bgr[0]; color[:, :, 1] = m*color_bgr[1]; color[:, :, 2] = m*color_bgr[2]
    return cv2.addWeighted(base, 1.0, color, alpha, 0)

def map_mask_to_image(root_dir: Path, masks_dir: Path, mask_path: Path) -> Path | None:
    rel=mask_path.relative_to(masks_dir)
    stem=mask_path.stem
    cands=[]
    if stem.endswith("_vote_mask"): cands.append(stem[:-11])
    if stem.endswith("_mask"):      cands.append(stem[:-5])
    cands=list(dict.fromkeys(cands))
    for base in cands:
        for ext in IMG_EXTS:
            p=(root_dir / rel.parent / f"{base}{ext}")
            if p.exists(): return p
    return None

def run_overlay(root_dir: Path, masks_dir: Path, out_dir: Path, n: int = 5, alpha: float = 0.45):
    masks_dir, root_dir, out_dir=Path(masks_dir), Path(root_dir), Path(out_dir)
    all_masks=list(masks_dir.rglob("*_mask.png"))+list(masks_dir.rglob("*_vote_mask.png"))
    if not all_masks:
        print("not found *_mask.png / *_vote_mask.png."); return
    random.seed(0)
    samples=random.sample(all_masks, min(n, len(all_masks)))
    saved=0
    for mpath in samples:
        ipath=map_mask_to_image(root_dir, masks_dir, mpath)
        if ipath is None or (not ipath.exists()): continue
        img=read_gray(ipath); msk=(read_gray(mpath)>0).astype(np.uint8)
        vis=overlay_bgr(img, msk, alpha=alpha, color_bgr=(255,0,0))
        out_path=out_dir/mpath.relative_to(masks_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path.with_suffix(".png")), vis); saved+=1
    print(f"Overlay saved: {saved} imagini în {out_dir}")

def iou_dice(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None=None):
    a=(a>0).astype(np.uint8)
    b=(b>0).astype(np.uint8)
    if valid is None:
        valid=np.ones_like(a, dtype=bool)
    a=a[valid]; b=b[valid]
    n_valid=int(a.size)
    sa=int(a.sum()); sb=int(b.sum())
    inter=int((a&b).sum())
    union=int((a|b).sum())
    if union==0:
        iou=1.0 if (sa==0 and sb==0) else 0.0
    else:
        iou=inter/union
    denom=sa+sb
    dice=1.0 if denom==0 else (2.0*inter/denom)
    return iou, dice, inter, union, sa, sb, n_valid

def _eval_one(root_dir, a_dir, b_dir, fid, sub, suff_a, suff_b, ignore_nodata):
    try:
        src=find_image(root_dir, sub, fid)
        if src is None:
            return ("miss", fid, sub, None)
        ipath=src
        a_path=Path(a_dir)/Path(sub)/f"{Path(ipath).stem}{suff_a}"
        b_path=Path(b_dir)/Path(sub)/f"{Path(ipath).stem}{suff_b}"
        if (not a_path.exists()) or (not b_path.exists()):
            return ("miss", fid, sub, None)
        a=(cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)>0).astype(np.uint8)
        b=(cv2.imread(str(b_path), cv2.IMREAD_GRAYSCALE)>0).astype(np.uint8)

        if ignore_nodata:
            img=read_gray(ipath)
            nodata=nodata_from_border_zeros(img)
            valid=(~nodata)
        else:
            valid=np.ones_like(a, dtype=bool)

        iou, dice, inter, union, sa, sb, n_valid = iou_dice(a, b, valid=valid)
        return ("ok", fid, sub, (iou, dice, inter, union, sa, sb, n_valid))
    except Exception:
        return ("fail", fid, sub, None)

def run_eval(root_dir: Path, meta_path: Path, a_dir: Path, b_dir: Path, out_csv: Path,
             suff_a="_mask.png", suff_b="_final_mask.png", ignore_nodata=True,
             workers=1):
    import csv
    df=load_meta(meta_path)
    root_dir, a_dir, b_dir, out_csv = Path(root_dir), Path(a_dir), Path(b_dir), Path(out_csv)
    rows=[(str(r._asdict()[{k.lower(): k for k in r._asdict().keys()}["id"]]).strip(),
             str(r._asdict()[{k.lower(): k for k in r._asdict().keys()}["path"]]).strip())
            for r in df.itertuples(index=False)]

    results=[]
    ok=miss=fail=0
    if workers and workers>1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut=[ex.submit(_eval_one, root_dir, a_dir, b_dir, fid, sub, suff_a, suff_b, ignore_nodata)
                   for fid, sub in rows]
            for f in tqdm(as_completed(fut), total=len(fut), desc="eval (paralel)"):
                status, fid, sub, val=f.result()
                if status=="ok": ok+=1; results.append((fid, sub, *val))
                elif status=="miss": miss+=1
                else: fail+=1
    else:
        for fid, sub in tqdm(rows, total=len(rows), desc="eval"):
            status, fid, sub, val=_eval_one(root_dir, a_dir, b_dir, fid, sub, suff_a, suff_b, ignore_nodata)
            if status=="ok": ok+=1; results.append((fid, sub, *val))
            elif status=="miss": miss+=1
            else: fail+=1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["id","path","IoU","Dice","Intersection","Union","SEA_A","SEA_B","ValidPixels"])
        for r in results:
            w.writerow(r)

    if results:
        ious=[r[2] for r in results]
        dices=[r[3] for r in results]
        mean_iou=float(np.mean(ious))
        mean_dice=float(np.mean(dices))
        print(f"\nEval done. pairs={len(results)}  missed={miss}  failed={fail}")
        print(f"Mean IoU:  {mean_iou:.4f}")
        print(f"Mean Dice: {mean_dice:.4f}")
    else:
        print("\nEval done, dar nu s a  potrivit niciun fisier .")




def build_parser():
    p=argparse.ArgumentParser(description="Segmentare SAR sea/land (LGMM & Voting) + overlay.")
    sub=p.add_subparsers(dest="cmd", required=True)

    p_lg=sub.add_parser("seg-lgmm")
    p_lg.add_argument("--root", required=True)
    p_lg.add_argument("--meta", required=True)
    p_lg.add_argument("--out",  required=True)
    p_lg.add_argument("--close", type=int, default=7)
    p_lg.add_argument("--min-area-sea", type=int, default=300)
    p_lg.add_argument("--min-area-land", type=int, default=0)
    p_lg.add_argument("--hole-sea", type=int, default=1200)
    p_lg.add_argument("--hole-land", type=int, default=300)
    p_lg.add_argument("--trim-top", type=float, default=1.0)
    p_lg.add_argument("--cov", type=str, default="diag", choices=["full","diag","tied","spherical"])
    p_lg.add_argument("--seed", type=int, default=0)
    p_lg.add_argument("--n-init", type=int, default=1)
    p_lg.add_argument("--bic-margin", type=float, default=-5.0)
    p_lg.add_argument("--all-sea", type=float, default=0.90)
    p_lg.add_argument("--all-land", type=float, default=0.05)
    p_lg.add_argument("--fit-max", type=int, default=50_000)
    p_lg.add_argument("--max-iter", type=int, default=50)
    p_lg.add_argument("--tol", type=float, default=1e-3)
    p_lg.add_argument("--reg-covar", type=float, default=1e-6)
    p_lg.add_argument("--pre-gauss", type=float, default=0.0, help="σ Gaussian pe log (0=off)")
    p_lg.add_argument("--workers", type=int, default=1)
    p_lg.add_argument("--dry-run", action="store_true")

    p_v=sub.add_parser("seg-vote")
    p_v.add_argument("--root", required=True)
    p_v.add_argument("--meta", required=True)
    p_v.add_argument("--out",  required=True)
    p_v.add_argument("--win", type=int, default=7)
    p_v.add_argument("--close", type=int, default=7)
    p_v.add_argument("--min-area-sea", type=int, default=300)
    p_v.add_argument("--min-area-land", type=int, default=0)
    p_v.add_argument("--kmeans-samples", type=int, default=100_000)
    p_v.add_argument("--workers", type=int, default=1)
    p_v.add_argument("--dry-run", action="store_true")

    p_ov=sub.add_parser("overlay")
    p_ov.add_argument("--root", required=True)
    p_ov.add_argument("--masks", required=True)
    p_ov.add_argument("--out", required=True)
    p_ov.add_argument("--n", type=int, default=5)
    p_ov.add_argument("--alpha", type=float, default=0.45)

    p_f=sub.add_parser("fuse", help="Majority vote între masca LGMM salvată și (KMeans, Otsu, Var) recalculat.")
    p_f.add_argument("--root",  required=True, help="Director imagini (ex: preprocessed/)")
    p_f.add_argument("--meta",  required=True, help="meta.csv/xlsx")
    p_f.add_argument("--gmm",   required=True, help="Director cu măști LGMM (ex: masks_gmm/)")
    p_f.add_argument("--out",   required=True, help="Director măști finale (ex: masks_final/)")
    p_f.add_argument("--win",   type=int, default=9, help="Fereastră pentru var_log")
    p_f.add_argument("--close", type=int, default=9, help="Kernel closing")
    p_f.add_argument("--min-area-sea",  type=int, default=400)
    p_f.add_argument("--min-area-land", type=int, default=200)
    p_f.add_argument("--hole-sea",  type=int, default=1500)
    p_f.add_argument("--hole-land", type=int, default=500)
    p_f.add_argument("--kmeans-samples", type=int, default=100000)
    p_f.add_argument("--thresh", type=int, default=3, help="Voturi minime pt SEA (din 4)")
    p_f.add_argument("--workers", type=int, default=1)
    p_f.add_argument("--dry-run", action="store_true")

    p_ev=sub.add_parser("eval", help="Compară două seturi de măști cu IoU/Dice (SEA=1).")
    p_ev.add_argument("--root", required=True, help="Director imagini (ex: preprocessed/)")
    p_ev.add_argument("--meta", required=True, help="meta.csv/xlsx")
    p_ev.add_argument("--a", required=True, help="Director măști A (ex: masks_gmm/)")
    p_ev.add_argument("--b", required=True, help="Director măști B (ex: masks_final/)")
    p_ev.add_argument("--suffix-a", default="_mask.png", help="Sufix fișiere A (implic.: _mask.png)")
    p_ev.add_argument("--suffix-b", default="_final_mask.png", help="Sufix fișiere B (implic.: _final_mask.png)")
    p_ev.add_argument("--out", required=True, help="CSV de ieșire (ex: eval\\lgmm_vs_final.csv)")
    p_ev.add_argument("--no-ignore-nodata", action="store_true", help="Nu exclude NoData din margini")
    p_ev.add_argument("--workers", type=int, default=1, help="Procese paralele")



    return p

def _fuse_one(root_dir, gmm_dir, out_root, fid, sub,
              win, close_k, min_area_sea, min_area_land,
              hole_sea, hole_land, kmeans_samples, thresh):
    try:
        src=find_image(root_dir, sub, fid)
        if src is None:
            return "miss"
        img=read_gray(src)

        nodata=nodata_from_border_zeros(img)
        x_log=to_log(img)
        valid=(~nodata)&np.isfinite(x_log)

        gmm_path=Path(gmm_dir)/Path(sub)/f"{Path(src).stem}_mask.png"
        if not gmm_path.exists():
            return "miss"
        m_gmm=(read_gray(gmm_path)>0).astype(np.uint8)
        m_gmm[~valid]=0  # NoData = LAND

        m_km =mask_kmeans_log(x_log, max_samples=kmeans_samples)
        m_ots=otsu_mask_from_log(x_log, valid=valid)
        m_var=mask_varlog_otsu(x_log, win=win)

        stack=np.stack([m_gmm, m_km, m_ots, m_var], axis=0).astype(np.uint8)
        fused=(stack.sum(axis=0) >= int(thresh)).astype(np.uint8)

        fused[nodata]=0
        mask=postprocess(fused, close_k=close_k,
                           min_area_sea=min_area_sea, min_area_land=min_area_land,
                           hole_sea=hole_sea, hole_land=hole_land)

        dst=out_root/Path(sub)/f"{Path(src).stem}_final_mask.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), (mask * 255).astype(np.uint8))
        return "ok"
    except Exception:
        return "fail"


def run_fuse(root_dir: Path, meta_path: Path, gmm_dir: Path, out_root: Path,
             win=9, close_k=9, min_area_sea=400, min_area_land=200,
             hole_sea=1500, hole_land=500, kmeans_samples=100_000,
             thresh=3, workers=1, dry_run=False):
    df=load_meta(meta_path)
    root_dir, out_root, gmm_dir = Path(root_dir), Path(out_root), Path(gmm_dir)
    rows=[(str(r._asdict()[{k.lower(): k for k in r._asdict().keys()}["id"]]).strip(),
             str(r._asdict()[{k.lower(): k for k in r._asdict().keys()}["path"]]).strip())
            for r in df.itertuples(index=False)]
    if dry_run:
        print(f"[dry-run] ar procesa {len(rows)} imagini."); return

    ok=miss=fail=0
    if workers and workers>1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut=[ex.submit(_fuse_one, root_dir, gmm_dir, out_root, fid, sub,
                             win, close_k, min_area_sea, min_area_land,
                             hole_sea, hole_land, kmeans_samples, thresh)
                   for fid, sub in rows]
            for f in tqdm(as_completed(fut), total=len(fut), desc="fuse (paralel)"):
                res=f.result(); ok+=(res=="ok"); miss+=(res=="miss"); fail+=(res=="fail")
    else:
        for fid, sub in tqdm(rows, total=len(rows), desc="fuse"):
            res=_fuse_one(root_dir, gmm_dir, out_root, fid, sub,
                            win, close_k, min_area_sea, min_area_land,
                            hole_sea, hole_land, kmeans_samples, thresh)
            ok+=(res=="ok"); miss+=(res=="miss"); fail+=(res=="fail")
    print(f"\nFuse done. saved={ok}  missed={miss}  failed={fail}")


def main():
    args=build_parser().parse_args()
    if args.cmd=="seg-lgmm":
        run_seg_lgmm(root_dir=Path(args.root), meta_path=Path(args.meta), out_root=Path(args.out),
                    close_k=args.close, min_area_sea=args.min_area_sea, min_area_land=args.min_area_land,
                    hole_sea=args.hole_sea, hole_land=args.hole_land,
                    trim_top=args.trim_top, cov=args.cov, seed=args.seed, n_init=args.n_init,
                    bic_margin=args.bic_margin, all_sea=args.all_sea, all_land=args.all_land,
                    fit_max=args.fit_max, max_iter=args.max_iter, tol=args.tol, reg_covar=args.reg_covar,
                    pre_gauss=args.pre_gauss,
                    workers=args.workers, dry_run=args.dry_run)
    elif args.cmd=="seg-vote":
        run_seg_vote(root_dir=Path(args.root), meta_path=Path(args.meta), out_root=Path(args.out),
                     win=args.win, close_k=args.close,
                     min_area_sea=args.min_area_sea, min_area_land=args.min_area_land,
                     kmeans_samples=args.kmeans_samples,
                     workers=args.workers, dry_run=args.dry_run)
    elif args.cmd=="overlay":
        run_overlay(root_dir=Path(args.root), masks_dir=Path(args.masks),
                    out_dir=Path(args.out), n=args.n, alpha=args.alpha)  
    elif args.cmd=="fuse":
        run_fuse(root_dir=Path(args.root), meta_path=Path(args.meta),
                 gmm_dir=Path(args.gmm), out_root=Path(args.out),
                 win=args.win, close_k=args.close,
                 min_area_sea=args.min_area_sea, min_area_land=args.min_area_land,
                 hole_sea=args.hole_sea, hole_land=args.hole_land,
                 kmeans_samples=args.kmeans_samples, thresh=args.thresh,
                 workers=args.workers, dry_run=args.dry_run)
    elif args.cmd=="eval":
        run_eval(root_dir=Path(args.root), meta_path=Path(args.meta),
                 a_dir=Path(args.a), b_dir=Path(args.b),
                 out_csv=Path(args.out),
                 suff_a=args.suffix_a, suff_b=args.suffix_b,
                 ignore_nodata=(not args.no_ignore_nodata),
                 workers=args.workers)


    

if __name__=="__main__":
    main()
