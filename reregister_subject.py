"""
reregister_subject.py
=====================
Re-runs the ANTs SyN registration for a single subject using the
already-oriented T1 and segmentation files that exist in test_outputs/.

Usage:
    python webapp_v3/reregister_subject.py UCSF-PDGM-0139

This will:
1. Load the oriented T1 from test_outputs/<pid>/registration/<pid>_T1_oriented.nii.gz
2. Re-run ANTs registration to the SRI24 atlas
3. Overwrite the existing registration/seg_registered files in test_outputs/
4. Regenerate the registration_quality.png visualisation
5. Copy the new visualisation into webapp_v3/data/<pid>/

Requires: antspyx, nibabel, numpy, matplotlib, Pillow
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TEST_OUTPUTS = PROJECT_ROOT / 'test_outputs'
ATLAS_T1     = PROJECT_ROOT / 'atlas_registration' / 'sri24_anatomy_nifti' / 'sri24' / 'spgr.nii'
DATA_DIR     = Path(__file__).parent / 'data'


def reregister(pid: str):
    import ants
    import nibabel as nib

    reg_dir = TEST_OUTPUTS / pid / 'registration'
    viz_dir = TEST_OUTPUTS / pid / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    t1_path  = reg_dir / f'{pid}_T1_oriented.nii.gz'
    seg_path = reg_dir / f'{pid}_seg_oriented.nii.gz'

    if not t1_path.exists():
        print(f"ERROR: T1 oriented file not found: {t1_path}")
        sys.exit(1)
    if not seg_path.exists():
        print(f"WARNING: Segmentation not found: {seg_path}")
        seg_path = None
    if not ATLAS_T1.exists():
        print(f"ERROR: Atlas T1 not found: {ATLAS_T1}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Re-registering {pid}")
    print(f"  T1 input: {t1_path.name} ({t1_path.stat().st_size//1024} KB)")
    print(f"  Atlas:    {ATLAS_T1}")
    print(f"{'='*60}")

    # Load images
    print("\n[1/4] Loading images...")
    moving = ants.image_read(str(t1_path))
    fixed  = ants.image_read(str(ATLAS_T1))

    print(f"  Moving shape: {moving.shape}")
    print(f"  Fixed shape:  {fixed.shape}")

    # Run ANTs SyN registration
    print("\n[2/4] Running ANTs SyN registration (this takes ~10-20 minutes)...")
    result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='SyN',
        aff_metric='mattes',
        syn_metric='mattes',
        grad_step=0.1,
        flow_sigma=3.0,
        total_sigma=0.0,
        aff_iterations=(2100, 1200, 1200, 10),
        aff_shrink_factors=(6, 4, 2, 1),
        aff_smoothing_sigmas=(3, 2, 1, 0),
        syn_iterations=(40, 20, 0),
        syn_shrink_factors=(3, 2, 1),
        syn_smoothing_sigmas=(2, 1, 0),
        write_composite_transform=False,
        verbose=True,
    )

    # Save registered T1
    print("\n[3/4] Saving registered volumes...")
    t1_reg_path  = reg_dir / f'{pid}_T1_registered.nii.gz'
    ants.image_write(result['warpedmovout'], str(t1_reg_path))
    print(f"  Saved: {t1_reg_path.name}")

    # Apply transform to segmentation
    if seg_path is not None:
        print("  Applying transform to segmentation...")
        seg_ants = ants.image_read(str(seg_path))
        seg_reg  = ants.apply_transforms(
            fixed=fixed,
            moving=seg_ants,
            transformlist=result['fwdtransforms'],
            interpolator='nearestNeighbor',
        )
        seg_reg_path = reg_dir / f'{pid}_seg_registered.nii.gz'
        ants.image_write(seg_reg, str(seg_reg_path))
        print(f"  Saved: {seg_reg_path.name}")

    # Save transforms
    for src, dst_name in zip(
        result['fwdtransforms'],
        [f'{pid}_forward_transform.mat', f'{pid}_forward_affine.mat']
    ):
        import shutil
        dst = reg_dir / dst_name
        shutil.copy2(src, str(dst))
        print(f"  Saved transform: {dst.name}")

    # Generate registration quality visualisation
    print("\n[4/4] Generating registration quality visualisation...")
    _make_reg_viz(
        fixed_path=str(ATLAS_T1),
        moved_path=str(t1_reg_path),
        output_path=str(viz_dir / 'registration_quality.png'),
        pid=pid,
    )

    # Copy new viz into webapp_v3/data if it exists
    v3_viz = DATA_DIR / pid / 'registration_quality.png'
    if v3_viz.parent.exists():
        import shutil
        shutil.copy2(str(viz_dir / 'registration_quality.png'), str(v3_viz))
        print(f"  Copied to webapp_v3/data/{pid}/registration_quality.png")

    print(f"\n[OK] Re-registration complete for {pid}")


def _make_reg_viz(fixed_path: str, moved_path: str, output_path: str, pid: str):
    """Create a registration quality figure (axial/coronal/sagittal overlay)."""
    import nibabel as nib
    fixed_img = nib.load(fixed_path)
    moved_img = nib.load(moved_path)

    fx = np.squeeze(fixed_img.get_fdata())
    mv = np.squeeze(moved_img.get_fdata())

    def norm(v):
        mn, mx = v.min(), v.max()
        return (v - mn) / (mx - mn) if mx > mn else v

    fx = norm(fx)
    mv = norm(mv)

    # Pick middle slices
    ax = fx.shape[2] // 2
    co = fx.shape[1] // 2
    sa = fx.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor='#080d1a')
    plt.suptitle(f'Registration Quality — {pid}', color='white', fontsize=13, fontweight='bold', y=0.98)

    row_labels = ['Atlas (Fixed)', 'Registered T1 (Moved)']
    views = [
        (np.flipud(fx[:, :, ax]),  np.flipud(mv[:, :, ax]),  'Axial'),
        (np.flipud(fx[:, co, :]),  np.flipud(mv[:, co, :]),  'Coronal'),
        (np.flipud(fx[sa, :, :]),  np.flipud(mv[sa, :, :]),  'Sagittal'),
    ]

    for col, (fx_sl, mv_sl, view) in enumerate(views):
        for row, (sl, label) in enumerate([(fx_sl, row_labels[0]), (mv_sl, row_labels[1])]):
            ax_obj = axes[row][col]
            ax_obj.imshow(sl.T, cmap='gray', aspect='auto', vmin=0, vmax=1)
            ax_obj.set_facecolor('#080d1a')
            ax_obj.set_xticks([])
            ax_obj.set_yticks([])
            if row == 0:
                ax_obj.set_title(view, color='#60a5fa', fontsize=10, fontweight='600')
            if col == 0:
                ax_obj.set_ylabel(label, color='#8a93a8', fontsize=8, labelpad=4)
            for spine in ax_obj.spines.values():
                spine.set_edgecolor('#1c2840')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#080d1a')
    plt.close(fig)
    print(f"  Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Re-run ANTs registration for one subject')
    parser.add_argument('pid', help='Subject ID, e.g. UCSF-PDGM-0139')
    args = parser.parse_args()
    reregister(args.pid)
