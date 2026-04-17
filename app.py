"""
webapp_v4/app.py
Complete Flask backend for the static demo webapp.
Supports both /api/... and /mri/api/... routes.
"""

import re
import json
from pathlib import Path
from flask import Flask, jsonify, render_template, send_from_directory, request

# ─── Paths ─────────────────────────────────────────────────────────────────────
WEBAPP_DIR = Path(__file__).parent.resolve()
DATA_DIR = WEBAPP_DIR / 'data'
SLICES_DIR = WEBAPP_DIR / 'static' / 'slices'   # pre-baked PNG store

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

# ─── Slice dimension cache ─────────────────────────────────────────────────────
_dim_cache = {}


def _get_slice_dims(pid: str, subfolder: str = ''):
    """
    Count pre-baked PNGs to determine volume dimensions for a given subfolder.
    subfolder=''                       -> native space
    subfolder='registered_brain'       -> registered pure T1
    subfolder='registered_projection'  -> registered + green overlay
    """
    cache_key = f'{pid}/{subfolder}'
    if cache_key in _dim_cache:
        return _dim_cache[cache_key]

    dims = {}
    all_present = True

    for view in ('axial', 'coronal', 'sagittal'):
        if subfolder:
            view_dir = SLICES_DIR / pid / subfolder / view
        else:
            view_dir = SLICES_DIR / pid / view

        if view_dir.exists():
            count = len(list(view_dir.glob('*.png')))
            dims[view] = count
        else:
            all_present = False
            dims[view] = 0

    if all_present and all(v > 0 for v in dims.values()):
        _dim_cache[cache_key] = dims
    else:
        _dim_cache[cache_key] = None

    return _dim_cache[cache_key]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# ─── Pages ────────────────────────────────────────────────────────────────────



@app.route('/')
def demo():
    return render_template('index.html')


@app.route('/mri_dashboard')
def landing():
    return render_template('mri_dashboard.html')


# ─── API: subjects ────────────────────────────────────────────────────────────
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


# ─── API: subject details ─────────────────────────────────────────────────────
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

    has_nifti = _get_slice_dims(pid, '') is not None

    images = {}
    for key, fname in [
        ('orthoview', meta.get('orthoview_image')),
        ('segmentation', meta.get('seg_image')),
        ('registration', meta.get('reg_image')),
    ]:
        if fname and (subj_dir / fname).exists():
            images[key] = f'/image/{pid}/{fname}'

    viz_path = subj_dir / f'{pid}_analysis_viz.png'
    if viz_path.exists():
        images['analysis_viz'] = f'/image/{pid}/{pid}_analysis_viz.png'

    return jsonify({
        'pid': pid,
        'meta': meta,
        'report': report,
        'qa': qa,
        'images': images,
        'has_nifti': has_nifti,
    })


# ─── Report ───────────────────────────────────────────────────────────────────
@app.route('/report/<pid>')
@app.route('/report/<pid>/')
@app.route('/mri/report/<pid>')
@app.route('/mri/report/<pid>/')
def full_report(pid: str):
    if pid not in SUBJECT_ORDER:
        return "Subject not found", 404

    import datetime
    import html

    def esc(s):
        return html.escape(str(s)).replace('{', '&#123;').replace('}', '&#125;')

    def esc_raw(s):
        return html.escape(str(s))

    subj_dir = DATA_DIR / pid
    meta = load_json(subj_dir / 'metadata.json')
    report = load_json(subj_dir / 'report.json')

    region_rows = ""
    for reg, vol in meta.get('top_regions', []):
        region_rows += f"<tr><td>{esc_raw(reg)}</td><td>{vol:.2f}</td></tr>\n"

    exam_date = datetime.date.today().strftime("%B %d, %Y")

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
            <thead><tr><th>Primary Laterality</th><th>Dominant Lobe/Structure</th><th>Total Volume (mL)</th><th>Midline Cross</th></tr></thead>
            <tbody>
                <tr><td>{esc_raw(meta.get('hemisphere', 'Unknown').capitalize())}</td><td>{esc_raw(meta.get('primary_lobe', 'Unknown'))}</td><td>{meta.get('total_volume_ml', 0):.2f}</td><td>{"Yes" if meta.get('crosses_midline') else "No"}</td></tr>
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


# ─── API: slice info ──────────────────────────────────────────────────────────
@app.route('/api/slice_info/<pid>')
@app.route('/api/slice_info/<pid>/')
@app.route('/mri/api/slice_info/<pid>')
@app.route('/mri/api/slice_info/<pid>/')
def api_slice_info(pid: str):
    """
    Return slice counts from the pre-baked PNG directory structure.
    ?space=native|registered
    """
    if pid not in SUBJECT_ORDER:
        return jsonify({'error': 'Subject not found'}), 404

    space = request.args.get('space', 'native')
    subfolder = 'registered_brain' if space == 'registered' else ''
    dims = _get_slice_dims(pid, subfolder)

    if dims is None:
        return jsonify({'error': 'Pre-baked slices not found. Run prebake_slices.py first.'}), 404

    axial = dims['axial']
    coronal = dims['coronal']
    sagittal = dims['sagittal']

    return jsonify({
        'shape': [sagittal, coronal, axial],
        'axial_slices': axial,
        'coronal_slices': coronal,
        'sagittal_slices': sagittal,
        'mid_axial': axial // 2,
        'mid_coronal': coronal // 2,
        'mid_sagittal': sagittal // 2,
    })


# ─── API: slice image ─────────────────────────────────────────────────────────
@app.route('/api/slice/<pid>/<view>/<int:idx>')
@app.route('/api/slice/<pid>/<view>/<int:idx>/')
@app.route('/mri/api/slice/<pid>/<view>/<int:idx>')
@app.route('/mri/api/slice/<pid>/<view>/<int:idx>/')
def api_slice(pid: str, view: str, idx: int):
    """
    Serve a pre-baked MRI slice PNG directly from static/slices/.
    Query params:
      ?space=native|registered
      ?seg=0|1
      ?viz=multicolor|projection
    """
    if pid not in SUBJECT_ORDER:
        return 'Not found', 404
    if view not in ('axial', 'coronal', 'sagittal'):
        return 'Bad view', 400

    space = request.args.get('space', 'native')
    show_seg = request.args.get('seg', '1') != '0'
    overlay_mode = request.args.get('viz', 'multicolor')

    if space == 'registered':
        if not show_seg:
            subfolder = 'registered_brain'
        else:
            if overlay_mode == 'projection':
                subfolder = 'registered_projection'
            else:
                subfolder = 'registered_projection'
    else:
        subfolder = ''

    if subfolder:
        png_path = SLICES_DIR / pid / subfolder / view / f'{idx}.png'
    else:
        png_path = SLICES_DIR / pid / view / f'{idx}.png'

    if not png_path.exists():
        return f'Slice not found — run prebake_slices.py (subfolder: {subfolder or "native"})', 404

    return send_from_directory(
        png_path.parent,
        png_path.name,
        mimetype='image/png',
        max_age=86400,
    )


# ─── API: QA ──────────────────────────────────────────────────────────────────
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


# ─── Image serving ────────────────────────────────────────────────────────────
@app.route('/image/<pid>/<filename>')
def serve_image(pid: str, filename: str):
    if pid not in SUBJECT_ORDER:
        return 'Not found', 404
    if not re.match(r'^[\w\-\.]+$', filename):
        return 'Forbidden', 403
    return send_from_directory(DATA_DIR / pid, filename)


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Webapp v4 -- Static Pre-Baked Demo Mode")
    print("  http://localhost:5001")
    print("=" * 60)

    ready = 0
    for pid in SUBJECT_ORDER:
        meta_path = DATA_DIR / pid / 'metadata.json'
        dims = _get_slice_dims(pid)
        tag = '[OK]' if meta_path.exists() else '[MISS]'
        bake_tag = f'slices:baked({dims['axial']}ax)' if dims else 'slices:NOT BAKED'
        print(f"  {tag} {pid}  ({bake_tag})")
        if meta_path.exists():
            ready += 1

    print(f"\n  {ready}/{len(SUBJECT_ORDER)} subjects ready")
    if ready == 0:
        print("\n  Run: python prepare_v3_data.py --dry-run")
    if not any(_get_slice_dims(pid) for pid in SUBJECT_ORDER):
        print("\n  !! No pre-baked slices found.")
        print("  Run: python prebake_slices.py")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5001, debug=False)