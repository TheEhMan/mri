"""
webapp_v3/app.py
Lean Flask backend for the hardcoded demo webapp.
No pipeline, no sessions — serves pre-generated data + live NIfTI slices.
"""

import io
import re
import json
import html
import datetime
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image
from flask import Flask, jsonify, render_template, send_from_directory, Response, request

# ─── Paths ─────────────────────────────────────────────────────────────────────
WEBAPP_DIR   = Path(__file__).parent.resolve()
DATA_DIR     = WEBAPP_DIR / 'data'
TEST_OUTPUTS = WEBAPP_DIR / 'test_output'

app = Flask(
    __name__,
    template_folder=str(WEBAPP_DIR / 'templates'),
    static_folder=str(WEBAPP_DIR / 'static'),
)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ─── Subject list ──────────────────────────────────────────────────────────────
SUBJECT_ORDER = [
    'UCSF-PDGM-0254',
    'UCSF-PDGM-0096',
    'UCSF-PDGM-0133',
    'UCSF-PDGM-0249',
    'UCSF-PDGM-0044',
]
ACCURACY_MAP = {
    'UCSF-PDGM-0254': 91.2,
    'UCSF-PDGM-0096': 91.2,
    'UCSF-PDGM-0133': 88.2,
    'UCSF-PDGM-0249': 88.2,
    'UCSF-PDGM-0044': 88.2,
}

# ─── NIfTI volume cache ───────────────────────────────────────────────────────
_vol_cache = {}


def _load_vol(pid: str, kind: str, space: str = 'native'):
    """
    Load and cache a NIfTI volume.
    space='native'     -> _T1_oriented.nii.gz  / _seg_oriented.nii.gz
    space='registered' -> _T1_registered.nii.gz / _seg_registered.nii.gz
    """
    key = f'{pid}_{kind}_{space}'
    if key not in _vol_cache:
        reg_dir = TEST_OUTPUTS / pid / 'registration'
        if space == 'registered':
            fname = f'{pid}_T1_registered.nii.gz' if kind == 't1' else f'{pid}_seg_registered.nii.gz'
        else:
            fname = f'{pid}_T1_oriented.nii.gz' if kind == 't1' else f'{pid}_seg_oriented.nii.gz'

        path = reg_dir / fname
        if path.exists():
            img = nib.load(str(path))
            _vol_cache[key] = (np.squeeze(img.get_fdata()), img.header.get_zooms()[:3])
        else:
            _vol_cache[key] = None
    return _vol_cache.get(key)


def _render_slice(t1_arr, voxel_sz, seg_arr, view: str, idx: int,
                  overlay_mode: str = 'multicolor') -> bytes:
    """
    Extract one slice, overlay segmentation, return PNG bytes.
    overlay_mode='multicolor' -> NCR=red, ED=yellow, ET=cyan
    overlay_mode='projection' -> all labels = bright green
    """
    if view == 'axial':
        idx = max(0, min(idx, t1_arr.shape[2] - 1))
        sl_t1 = t1_arr[:, :, idx].T
        sl_seg = seg_arr[:, :, idx].T if seg_arr is not None else None
        asp = voxel_sz[1] / voxel_sz[0]
    elif view == 'coronal':
        idx = max(0, min(idx, t1_arr.shape[1] - 1))
        sl_t1 = t1_arr[:, idx, :].T
        sl_seg = seg_arr[:, idx, :].T if seg_arr is not None else None
        asp = voxel_sz[2] / voxel_sz[0]
    else:  # sagittal
        idx = max(0, min(idx, t1_arr.shape[0] - 1))
        sl_t1 = t1_arr[idx, :, :].T
        sl_seg = seg_arr[idx, :, :].T if seg_arr is not None else None
        asp = voxel_sz[2] / voxel_sz[1]

    nonzero = t1_arr[t1_arr > 0]
    if nonzero.size > 0:
        lo, hi = np.percentile(nonzero, (1, 99))
    else:
        lo, hi = 0, 1

    sl_t1 = np.clip((sl_t1 - lo) / max(hi - lo, 1e-6), 0, 1)
    gray = (sl_t1 * 255).astype(np.uint8)

    gray = np.flipud(gray)
    if sl_seg is not None:
        sl_seg = np.flipud(sl_seg)

    rgb = np.stack([gray, gray, gray], axis=-1)

    if sl_seg is not None:
        if overlay_mode == 'projection':
            mask = sl_seg.astype(np.int32) > 0
            alpha = 0.62
            r, g, b = 50, 220, 80
            if mask.any():
                rgb[mask, 0] = np.clip(rgb[mask, 0] * (1 - alpha) + r * alpha, 0, 255).astype(np.uint8)
                rgb[mask, 1] = np.clip(rgb[mask, 1] * (1 - alpha) + g * alpha, 0, 255).astype(np.uint8)
                rgb[mask, 2] = np.clip(rgb[mask, 2] * (1 - alpha) + b * alpha, 0, 255).astype(np.uint8)
        else:
            seg_colors = {
                1: (220, 55, 55, 0.55),   # NCR
                2: (240, 190, 40, 0.50),  # ED
                3: (40, 185, 220, 0.55),  # ET
            }
            for label, (r, g, b, alpha) in seg_colors.items():
                mask = sl_seg.astype(np.int32) == label
                if not mask.any():
                    continue
                rgb[mask, 0] = np.clip(rgb[mask, 0] * (1 - alpha) + r * alpha, 0, 255).astype(np.uint8)
                rgb[mask, 1] = np.clip(rgb[mask, 1] * (1 - alpha) + g * alpha, 0, 255).astype(np.uint8)
                rgb[mask, 2] = np.clip(rgb[mask, 2] * (1 - alpha) + b * alpha, 0, 255).astype(np.uint8)

    target = 480
    if asp > 1:
        new_h, new_w = int(target * asp), target
    else:
        new_h, new_w = target, int(target / max(asp, 0.1))

    pil = Image.fromarray(rgb.astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    pil.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def esc(s):
    return html.escape(str(s)).replace('{', '&#123;').replace('}', '&#125;')


def esc_raw(s):
    return html.escape(str(s))


# ─── Page routes ──────────────────────────────────────────────────────────────
@app.route('/')
@app.route('/mri-dashboard')
def landing():
    return render_template('mri_dashboard.html')


@app.route('/demo')
def demo():
    return render_template('index.html')


# ─── API routes ───────────────────────────────────────────────────────────────
@app.route('/api/subjects')
@app.route('/api/subjects/')
@app.route('/mri/api/subjects')
@app.route('/mri/api/subjects/')
def api_subjects():
    subjects = []
    for pid in SUBJECT_ORDER:
        meta_path = DATA_DIR / pid / 'metadata.json'
        meta = load_json(meta_path)

        subjects.append({
            'pid': pid,
            'accuracy': ACCURACY_MAP.get(pid, 0),
            'ready': meta_path.exists(),
            'total_volume': meta.get('total_volume_ml', '—'),
            'hemisphere': meta.get('hemisphere', '—').capitalize(),
            'primary_lobe': meta.get('primary_lobe', '—'),
        })
    return jsonify(subjects)


@app.route('/api/subject/<pid>')
@app.route('/api/subject/<pid>/')
@app.route('/mri/api/subject/<pid>')
@app.route('/mri/api/subject/<pid>/')
def api_subject(pid: str):
    if pid not in SUBJECT_ORDER:
        return jsonify({'error': 'Subject not found'}), 404

    subj_dir = DATA_DIR / pid
    meta = load_json(subj_dir / 'metadata.json')
    report = load_json(subj_dir / 'report.json')
    qa = load_json(subj_dir / 'qa.json')

    reg_dir = TEST_OUTPUTS / pid / 'registration'
    has_nifti = (reg_dir / f'{pid}_T1_registered.nii.gz').exists()

    images = {}
    for key, fname in [
        ('orthoview', meta.get('orthoview_image')),
        ('segmentation', meta.get('seg_image')),
        ('registration', meta.get('reg_image')),
    ]:
        if fname and (subj_dir / fname).exists():
            images[key] = f'/image/{pid}/{fname}'

    viz_path = TEST_OUTPUTS / pid / 'medgemma_reports' / f'{pid}_analysis_viz.png'
    if viz_path.exists():
        images['analysis_viz'] = f'/image_ext/{pid}/medgemma_reports/{pid}_analysis_viz.png'

    return jsonify({
        'pid': pid,
        'meta': meta,
        'report': report,
        'qa': qa,
        'images': images,
        'has_nifti': has_nifti,
    })


@app.route('/report/<pid>')
@app.route('/report/<pid>/')
@app.route('/mri/report/<pid>')
@app.route('/mri/report/<pid>/')
def full_report(pid: str):
    if pid not in SUBJECT_ORDER:
        return "Subject not found", 404

    subj_dir = DATA_DIR / pid
    meta = load_json(subj_dir / 'metadata.json')
    report = load_json(subj_dir / 'report.json')

    region_rows = ""
    for reg, vol in meta.get('top_regions', []):
        try:
            vol_str = f"{float(vol):.2f}"
        except Exception:
            vol_str = esc_raw(vol)
        region_rows += f"<tr><td>{esc_raw(reg)}</td><td>{vol_str}</td></tr>\n"

    exam_date = datetime.date.today().strftime("%B %d, %Y")

    total_volume = meta.get('total_volume_ml', 0)
    try:
        total_volume_str = f"{float(total_volume):.2f}"
    except Exception:
        total_volume_str = esc_raw(total_volume)

    report_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>MedGemma Report - {esc_raw(pid)}</title>
        <style>
            body {{ background-color: #f8f9fa; padding: 20px; margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            .report-wrapper {{ line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; background-color: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #0056b3; text-align: left; }}
            .report-wrapper h1 {{ color: #0056b3; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 0; margin-bottom: 20px; font-size: 2rem; }}
            .report-wrapper h2 {{ color: #444; font-size: 1.2rem; margin-top: 30px; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            .meta-info {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; background-color: #f1f5f9; padding: 15px; border-radius: 5px; margin-bottom: 25px; font-size: 0.95rem; }}
            .meta-info p {{ margin: 0; color: #333; }}
            .meta-info strong {{ color: #222; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.95rem; background-color: #fff; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; color: #333; }}
            th {{ background-color: #f8f9fa; font-weight: 600; color: #222; }}
            .assessment-section {{ margin-bottom: 20px; }}
            .assessment-content {{ background-color: #fafafa; padding: 15px; border-left: 4px solid #0056b3; border-radius: 0 4px 4px 0; color: #333; }}
            .assessment-content p {{ margin-top: 0; margin-bottom: 0.5rem; }}
            .assessment-content p:last-child {{ margin-bottom: 0; }}
            .footer {{ margin-top: 40px; font-size: 0.85rem; color: #777; text-align: center; border-top: 1px solid #eee; padding-top: 15px; }}
        </style>
    </head>
    <body>
    <div class="report-wrapper">
        <h1>MedGemma Multimodal MRI Report</h1>
        <div class="meta-info">
            <p><strong>Patient ID:</strong> {esc_raw(pid)}</p>
            <p><strong>Date of Exam:</strong> {esc_raw(exam_date)}</p>
            <p><strong>Sequences Evaluated:</strong> T1, T1CE, T2, FLAIR</p>
        </div>

        <h2>1. Deterministic Volumetric &amp; Spatial Analysis (Truth Anchor)</h2>
        <table>
            <thead>
                <tr>
                    <th>Primary Laterality</th>
                    <th>Dominant Lobe/Structure</th>
                    <th>Total Volume (mL)</th>
                    <th>Midline Cross</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>{esc_raw(meta.get('hemisphere', 'Unknown').capitalize())}</td>
                    <td>{esc_raw(meta.get('primary_lobe', 'Unknown'))}</td>
                    <td>{total_volume_str}</td>
                    <td>{"Yes" if meta.get('crosses_midline') else "No"}</td>
                </tr>
            </tbody>
        </table>

        <table>
            <thead><tr><th>Region (AAL Atlas)</th><th>Overlap Volume (mL)</th></tr></thead>
            <tbody>{region_rows}</tbody>
        </table>

        <h2>2. Morphological Characteristic Assessment</h2>
        <div class="assessment-section"><div class="assessment-content">
            <p>{esc(report.get('morphology', 'No data available.'))}</p>
        </div></div>

        <h2>3. Anatomical Relationship Assessment</h2>
        <div class="assessment-section"><div class="assessment-content">
            <p>{esc(report.get('anatomy', 'No data available.'))}</p>
        </div></div>

        <h2>4. Factual Correlation Assessment</h2>
        <div class="assessment-section"><div class="assessment-content">
            <p>{esc(report.get('correlation', 'No data available.'))}</p>
        </div></div>

        <div class="footer">
            <p><strong>Note:</strong> This report was generated using the MedGemma 1.5 pipeline leveraging deterministic Cost Function Masking (CFM) registration and symbolic anatomical grounding. Clinical correlation is required.</p>
        </div>
    </div>
    </body>
    </html>
    """
    return report_html


@app.route('/api/slice_info/<pid>')
@app.route('/api/slice_info/<pid>/')
@app.route('/mri/api/slice_info/<pid>')
@app.route('/mri/api/slice_info/<pid>/')
def api_slice_info(pid: str):
    if pid not in SUBJECT_ORDER:
        return jsonify({'error': 'Subject not found'}), 404

    space = request.args.get('space', 'native')
    res = _load_vol(pid, 't1', space)
    if res is None:
        return jsonify({'error': 'NIfTI not found'}), 404

    arr, vox = res
    return jsonify({
        'shape': list(arr.shape),
        'axial_slices': arr.shape[2],
        'coronal_slices': arr.shape[1],
        'sagittal_slices': arr.shape[0],
        'mid_axial': arr.shape[2] // 2,
        'mid_coronal': arr.shape[1] // 2,
        'mid_sagittal': arr.shape[0] // 2,
    })


@app.route('/api/slice/<pid>/<view>/<int:idx>')
@app.route('/api/slice/<pid>/<view>/<int:idx>/')
@app.route('/mri/api/slice/<pid>/<view>/<int:idx>')
@app.route('/mri/api/slice/<pid>/<view>/<int:idx>/')
def api_slice(pid: str, view: str, idx: int):
    if pid not in SUBJECT_ORDER:
        return 'Not found', 404
    if view not in ('axial', 'coronal', 'sagittal'):
        return 'Bad view', 400

    space = request.args.get('space', 'native')
    show_seg = request.args.get('seg', '1') != '0'
    overlay_mode = request.args.get('viz', 'multicolor')

    t1_res = _load_vol(pid, 't1', space)
    seg_res = _load_vol(pid, 'seg', space) if show_seg else None

    if t1_res is None:
        return 'NIfTI not found', 404

    t1_arr, vox = t1_res
    seg_arr = seg_res[0] if seg_res else None

    png_bytes = _render_slice(t1_arr, vox, seg_arr, view, idx, overlay_mode)
    return Response(
        png_bytes,
        mimetype='image/png',
        headers={'Cache-Control': 'public, max-age=3600'},
    )


@app.route('/api/qa/<pid>/<q_id>')
@app.route('/api/qa/<pid>/<q_id>/')
@app.route('/mri/api/qa/<pid>/<q_id>')
@app.route('/mri/api/qa/<pid>/<q_id>/')
def api_qa(pid: str, q_id: str):
    if pid not in SUBJECT_ORDER:
        return jsonify({'error': 'Subject not found'}), 404

    qa = load_json(DATA_DIR / pid / 'qa.json')
    answer = qa.get(q_id)
    if answer is None:
        return jsonify({'error': f'Question {q_id} not found'}), 404
    return jsonify({'pid': pid, 'q_id': q_id, 'answer': answer})


# ─── Image routes ─────────────────────────────────────────────────────────────
@app.route('/image/<pid>/<filename>')
def serve_image(pid: str, filename: str):
    if pid not in SUBJECT_ORDER:
        return 'Not found', 404
    if not re.match(r'^[\w\-.]+$', filename):
        return 'Forbidden', 403
    return send_from_directory(DATA_DIR / pid, filename)


@app.route('/image_ext/<pid>/<path:subpath>')
def serve_image_ext(pid: str, subpath: str):
    if pid not in SUBJECT_ORDER:
        return 'Not found', 404
    full = TEST_OUTPUTS / pid / subpath
    if not full.exists():
        return 'Not found', 404
    return send_from_directory(full.parent, full.name)


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Webapp v3 -- Hardcoded Demo Mode")
    print("  http://localhost:5001")
    print("=" * 60)

    ready = 0
    for pid in SUBJECT_ORDER:
        meta_path = DATA_DIR / pid / 'metadata.json'
        nifti_ok = (TEST_OUTPUTS / pid / 'registration' / f'{pid}_T1_registered.nii.gz').exists()
        tag = '[OK]' if meta_path.exists() else '[MISS]'
        nii_tag = 'NIfTI:ok' if nifti_ok else 'NIfTI:missing'
        print(f"  {tag} {pid}  ({nii_tag})")
        if meta_path.exists():
            ready += 1

    print(f"\n  {ready}/{len(SUBJECT_ORDER)} subjects ready")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5001, debug=False)