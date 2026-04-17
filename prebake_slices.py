"""
webapp_v4/prebake_slices.py
============================
Run this script ONCE, locally, before building the Docker image.

It pre-renders every MRI slice for every subject into static PNG files
stored under webapp_v4/static/slices/<pid>/<view>/<idx>.png

This replaces the dynamic NIfTI-to-PNG pipeline in app.py so the
production container needs no nibabel/numpy at all — it simply serves
pre-rendered static files.

Usage:
    python webapp_v4/prebake_slices.py
"""

import sys
import io
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
WEBAPP_DIR   = Path(__file__).parent.resolve()
TEST_OUTPUTS = WEBAPP_DIR / 'test_output'
SLICES_DIR   = WEBAPP_DIR / 'static' / 'slices'

SUBJECT_ORDER = [
    'UCSF-PDGM-0254',
    'UCSF-PDGM-0096',
    'UCSF-PDGM-0133',
    'UCSF-PDGM-0249',
    'UCSF-PDGM-0044',
]

VIEWS = ['axial', 'coronal', 'sagittal']


def _load_nifti(pid: str, kind: str, space: str = 'native'):
    reg_dir = TEST_OUTPUTS / pid / 'registration'
    if space == 'registered':
        fname = f'{pid}_T1_registered.nii.gz' if kind == 't1' else f'{pid}_seg_registered.nii.gz'
    else:
        fname = f'{pid}_T1_oriented.nii.gz' if kind == 't1' else f'{pid}_seg_oriented.nii.gz'
    path = reg_dir / fname
    if not path.exists():
        return None, None
    img = nib.load(str(path))
    return np.squeeze(img.get_fdata()), img.header.get_zooms()[:3]


def _render_slice(t1_arr, voxel_sz, seg_arr, view: str, idx: int,
                  overlay_mode: str = 'multicolor') -> bytes:
    """Render one MRI slice with segmentation overlay and return PNG bytes.
    overlay_mode='multicolor' -> NCR=red, ED=yellow, ET=cyan
    overlay_mode='projection' -> all labels = bright green
    """
    if view == 'axial':
        idx    = max(0, min(idx, t1_arr.shape[2] - 1))
        sl_t1  = t1_arr[:, :, idx].T
        sl_seg = seg_arr[:, :, idx].T if seg_arr is not None else None
        asp    = voxel_sz[1] / voxel_sz[0]
    elif view == 'coronal':
        idx    = max(0, min(idx, t1_arr.shape[1] - 1))
        sl_t1  = t1_arr[:, idx, :].T
        sl_seg = seg_arr[:, idx, :].T if seg_arr is not None else None
        asp    = voxel_sz[2] / voxel_sz[0]
    else:  # sagittal
        idx    = max(0, min(idx, t1_arr.shape[0] - 1))
        sl_t1  = t1_arr[idx, :, :].T
        sl_seg = seg_arr[idx, :, :].T if seg_arr is not None else None
        asp    = voxel_sz[2] / voxel_sz[1]

    lo, hi = np.percentile(t1_arr[t1_arr > 0], (1, 99)) if t1_arr.max() > 0 else (0, 1)
    sl_t1  = np.clip((sl_t1 - lo) / max(hi - lo, 1e-6), 0, 1)
    gray   = (sl_t1 * 255).astype(np.uint8)
    gray   = np.flipud(gray)
    if sl_seg is not None:
        sl_seg = np.flipud(sl_seg)

    rgb = np.stack([gray, gray, gray], axis=-1)

    if sl_seg is not None:
        if overlay_mode == 'projection':
            mask  = sl_seg.astype(np.int32) > 0
            alpha = 0.62
            r, g, b = 50, 220, 80   # bright green #32DC50
            if mask.any():
                rgb[mask, 0] = np.clip(rgb[mask, 0] * (1 - alpha) + r * alpha, 0, 255).astype(np.uint8)
                rgb[mask, 1] = np.clip(rgb[mask, 1] * (1 - alpha) + g * alpha, 0, 255).astype(np.uint8)
                rgb[mask, 2] = np.clip(rgb[mask, 2] * (1 - alpha) + b * alpha, 0, 255).astype(np.uint8)
        else:
            SEG_COLORS = {
                1: (220,  55,  55, 0.55),
                2: (240, 190,  40, 0.50),
                3: ( 40, 185, 220, 0.55),
            }
            for label, (r, g, b, alpha) in SEG_COLORS.items():
                mask = sl_seg.astype(np.int32) == label
                if not mask.any():
                    continue
                rgb[mask, 0] = np.clip(rgb[mask, 0] * (1 - alpha) + r * alpha, 0, 255).astype(np.uint8)
                rgb[mask, 1] = np.clip(rgb[mask, 1] * (1 - alpha) + g * alpha, 0, 255).astype(np.uint8)
                rgb[mask, 2] = np.clip(rgb[mask, 2] * (1 - alpha) + b * alpha, 0, 255).astype(np.uint8)

    TARGET = 480
    if asp > 1:
        new_h, new_w = int(TARGET * asp), TARGET
    else:
        new_h, new_w = TARGET, int(TARGET / max(asp, 0.1))

    pil = Image.fromarray(rgb.astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    pil.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def prebake(pid: str):
    print(f"\n  [{pid}]")

    # ── 1. NATIVE space: multicolor seg overlay ──────────────────────────────
    t1_arr, vox = _load_nifti(pid, 't1', 'native')
    seg_arr, _  = _load_nifti(pid, 'seg', 'native')
    if t1_arr is None:
        print(f"    !! Native NIfTI not found for {pid}, skipping.")
    else:
        shape = t1_arr.shape
        slice_counts = {'axial': shape[2], 'coronal': shape[1], 'sagittal': shape[0]}
        for view in VIEWS:
            out_dir = SLICES_DIR / pid / view
            out_dir.mkdir(parents=True, exist_ok=True)
            total = slice_counts[view]
            print(f"    [native/{view}]: {total} slices ... ", end='', flush=True)
            for idx in range(total):
                out_path = out_dir / f'{idx}.png'
                if out_path.exists():
                    continue
                png = _render_slice(t1_arr, vox, seg_arr, view, idx)
                out_path.write_bytes(png)
            print("done")

    # ── 2. REGISTERED space: pure T1, no seg (Co-registered Brain) ──────────
    t1_reg, vox_r = _load_nifti(pid, 't1', 'registered')
    seg_reg, _    = _load_nifti(pid, 'seg', 'registered')
    if t1_reg is None:
        print(f"    !! Registered NIfTI not found for {pid}, skipping registered panels.")
    else:
        shape_r = t1_reg.shape
        sc_r    = {'axial': shape_r[2], 'coronal': shape_r[1], 'sagittal': shape_r[0]}

        # 2a. No seg — pure T1
        for view in VIEWS:
            out_dir = SLICES_DIR / pid / 'registered_brain' / view
            out_dir.mkdir(parents=True, exist_ok=True)
            total = sc_r[view]
            print(f"    [registered_brain/{view}]: {total} slices ... ", end='', flush=True)
            for idx in range(total):
                out_path = out_dir / f'{idx}.png'
                if out_path.exists():
                    continue
                png = _render_slice(t1_reg, vox_r, None, view, idx)  # No seg mask
                out_path.write_bytes(png)
            print("done")

        # 2b. Green projection overlay (Tumor Projection)
        for view in VIEWS:
            out_dir = SLICES_DIR / pid / 'registered_projection' / view
            out_dir.mkdir(parents=True, exist_ok=True)
            total = sc_r[view]
            print(f"    [registered_projection/{view}]: {total} slices ... ", end='', flush=True)
            for idx in range(total):
                out_path = out_dir / f'{idx}.png'
                if out_path.exists():
                    continue
                png = _render_slice(t1_reg, vox_r, seg_reg, view, idx, overlay_mode='projection')
                out_path.write_bytes(png)
            print("done")


if __name__ == '__main__':
    print("=" * 60)
    print("  webapp_v4 Slice Pre-Baker")
    print("  Renders all NIfTI slices → static PNGs")
    print("=" * 60)

    for pid in SUBJECT_ORDER:
        prebake(pid)

    print("\n" + "=" * 60)
    print("  Done! All slices saved to static/slices/")
    print("  You can now build the Docker image.")
    print("=" * 60)
