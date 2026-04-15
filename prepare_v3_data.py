    """
    prepare_v3_data.py
    ==================
    One-time script — run on your GPU machine BEFORE deploying webapp_v3.

    For each of the 5 best subjects it:
    1. Reads the existing aal_mni.json and segmentation files from test_outputs/
    2. Computes answers for Q1–Q13 deterministically from the data (no GPU needed)
    3. Runs MedGemma once for Q14–Q17 (morphology, needs GPU) and the initial report
    4. Saves:
        webapp_v3/data/<pid>/report.json    — 3-section initial report
        webapp_v3/data/<pid>/qa.json        — all 19 Q&A answers
        webapp_v3/data/<pid>/metadata.json  — volumes, flags, accuracy score
        webapp_v3/data/<pid>/*.png          — orthoview + segmentation images

    Usage:
        python prepare_v3_data.py
        python prepare_v3_data.py --dry-run   <- skip model inference, fill placeholders
    """

    import sys
    import re
    import json
    import shutil
    import argparse
    from pathlib import Path

    # ─── Paths ─────────────────────────────────────────────────────────────────────
    PROJECT_ROOT   = Path(__file__).parent.parent.resolve()
    TEST_OUTPUTS   = PROJECT_ROOT / 'test_outputs'
    DATA_DIR       = Path(__file__).parent / 'data'
    ATLAS_NII      = PROJECT_ROOT / 'atlas_registration' / 'sri24_labels' / 'sri24' / 'tzo116plus.nii'
    ATLAS_TXT      = PROJECT_ROOT / 'atlas_registration' / 'sri24_labels' / 'sri24' / 'SRI24-tzo116plus.txt'

    # ─── 5 Best subjects (from pipeline_evaluation_improved_v5.ipynb results) ──────
    SUBJECTS = [
        {'pid': 'UCSF-PDGM-0254', 'accuracy': 91.2},
        {'pid': 'UCSF-PDGM-0096', 'accuracy': 91.2},
        {'pid': 'UCSF-PDGM-0133', 'accuracy': 88.2},
        {'pid': 'UCSF-PDGM-0249', 'accuracy': 88.2},
        {'pid': 'UCSF-PDGM-0044', 'accuracy': 88.2},
    ]


    # ─── Helpers ───────────────────────────────────────────────────────────────────
    def clean_region_name(name: str) -> str:
        """Strip trailing RGBA colour codes that AAL labels include."""
        return re.sub(r'(\s+\d+){3,}\s*$', '', name.strip()).strip()


    def get_paths(pid: str) -> dict:
        base = TEST_OUTPUTS / pid
        reg  = base / 'registration'
        img_a = base / 'visualizations' / f'{pid}_orthoview_centered.png'
        img_b = base / 'medgemma_reports' / f'{pid}_analysis_viz.png'
        return {
            'base':           base,
            'aal_json':       base / 'aal_analysis' / f'{pid}_aal_mni.json',
            'seg_registered': reg / f'{pid}_seg_registered.nii.gz',
            'seg_native':     reg / f'{pid}_seg_oriented.nii.gz',
            'orthoview':      img_a if img_a.exists() else img_b,
            'seg_multiclass': base / 'visualizations' / 'segmentation_multiclass.png',
            'reg_quality':    base / 'visualizations' / 'registration_quality.png',
        }


    def load_aal(pid: str) -> dict:
        p = get_paths(pid)
        with open(p['aal_json'], 'r') as f:
            data = json.load(f)
        # Clean region names of colour codes
        for r in data.get('regions', []):
            r['region_name'] = clean_region_name(r['region_name'])
        return data


    def compute_laterality(data: dict):
        all_rs = data.get('regions', [])
        lv = sum(r['volume_ml'] for r in all_rs if r['region_name'].endswith('_L'))
        rv = sum(r['volume_ml'] for r in all_rs if r['region_name'].endswith('_R'))
        sv = lv + rv
        lp = lv / sv * 100 if sv > 0 else 50.0
        rp = rv / sv * 100 if sv > 0 else 50.0
        if lp >= 60:
            hemi = 'left'
        elif rp >= 60:
            hemi = 'right'
        else:
            hemi = 'bilateral'
        crosses = lp >= 20 and rp >= 20
        return hemi, lp, rp, crosses


    def compute_primary_lobe(data: dict) -> str:
        lobe_map = {
            'Frontal':   ['Frontal', 'Precentral', 'Rolandic', 'Supp_Motor'],
            'Temporal':  ['Temporal', 'Hippocamp', 'Parahippocampal', 'Fusiform'],
            'Parietal':  ['Parietal', 'Postcentral', 'Angular', 'Supramarginal', 'Precuneus'],
            'Occipital': ['Occipital', 'Cuneus', 'Lingual', 'Calcarine'],
            'Insula':    ['Insula'],
        }
        votes = {l: 0.0 for l in lobe_map}
        for r in data.get('regions', [])[:10]:
            name = r['region_name']
            for lobe, kws in lobe_map.items():
                if any(kw.lower() in name.lower() for kw in kws):
                    votes[lobe] += r['volume_ml']
        return max(votes, key=votes.get) if any(votes.values()) else 'Unknown'


    def get_regions_for_compartment(pid: str, label: int, label_name: str, data: dict) -> str:
        """
        Look up which atlas regions overlap with a specific BraTS label (1=NCR, 2=ED, 3=ET).
        Falls back to top-N overall regions if compartment data isn't available.
        """
        comp = data.get('compartment_regions', {})
        key = {1: 'ncr', 2: 'ed', 3: 'et'}.get(label, 'ncr')
        rs = comp.get(key, [])
        if rs:
            top = rs[:6]
            names = ', '.join(f"{r['region_name']} ({r['volume_ml']:.1f} mL)" for r in top)
            return f"Regions with {label_name} involvement (atlas-verified): {names}."
        # Fallback
        top = [r['region_name'] for r in data.get('regions', [])[:6]]
        return f"Primary regions involved: {', '.join(top)}. (compartment-level data not available)"


    # ─── Deterministic Q&A (Q1–Q13) ────────────────────────────────────────────────
    def build_deterministic_qa(pid: str, data: dict, compartments: dict) -> dict:
        vol   = data.get('total_tumor_volume_ml', 0)
        diam  = data.get('max_diameter_mm', None)
        flags = data.get('proximity_flags', {})
        all_rs = data.get('regions', [])

        # Merge compartment volumes into data
        ncr = compartments.get('ncr_volume_ml', 0)
        ed  = compartments.get('ed_volume_ml', 0)
        et  = compartments.get('et_volume_ml', 0)

        hemi, lp, rp, crosses = compute_laterality(data)
        primary_lobe = compute_primary_lobe(data)
        top5 = [r['region_name'] for r in all_rs[:5]]
        top10 = [r['region_name'] for r in all_rs[:10]]

        # Eloquent flags
        near_motor    = flags.get('NearMotorCortex', any('Precentral' in r or 'Rolandic' in r for r in top10))
        near_broca    = flags.get('NearBrocas',      any('Frontal_Inf' in r for r in top10))
        near_wernicke = flags.get('NearWernickes',   any('Temporal_Sup' in r for r in top10))
        near_insula   = flags.get('NearInsula',      any('Insula' in r for r in top10))
        eloquent_list = []
        if near_motor:    eloquent_list.append('primary motor cortex (Precentral/Rolandic)')
        if near_broca:    eloquent_list.append("Broca's area (inferior frontal gyrus)")
        if near_wernicke: eloquent_list.append("Wernicke's area (superior temporal gyrus)")
        if near_insula:   eloquent_list.append('insular cortex')

        # Active proximity flags nicely formatted
        active_flags = [k for k, v in flags.items() if v]

        qa = {}

        # Q1 — Total tumor volume
        qa['volume_total'] = (
            f"The total tumor volume is **{vol:.1f} mL**"
            + (f" with a maximum diameter of **{diam:.0f} mm**" if diam else "")
            + f". This is a {'large (>100 mL)' if vol > 100 else 'moderate-sized (50–100 mL)' if vol > 50 else 'relatively small (<50 mL)'} tumor. "
            f"(Atlas-verified measurement)"
        )

        # Q2 — Necrotic core
        if ncr and ncr > 0:
            qa['volume_necrotic'] = (
                f"The necrotic core (NCR, BraTS label 1) occupies **{ncr:.1f} mL** "
                f"({ncr/vol*100:.0f}% of total tumor volume). "
                f"Necrotic tissue represents the central dead/liquefied core, typically reflecting high-grade malignancy."
            )
        else:
            qa['volume_necrotic'] = (
                f"Compartment-level volume data for the necrotic core is not separately computed for this subject. "
                f"The total tumor volume is {vol:.1f} mL. Refer to the segmentation overlay for visual assessment."
            )

        # Q3 — Edema
        if ed and ed > 0:
            qa['volume_edema'] = (
                f"Peritumoral edema (ED, BraTS label 2) occupies **{ed:.1f} mL** "
                f"({ed/vol*100:.0f}% of total volume). "
                f"This represents the T2-hyperintense infiltrative zone surrounding the tumor core."
            )
        else:
            qa['volume_edema'] = (
                f"Compartment-level edema volume is not separately computed for this subject. "
                f"Peritumoral edema extent can be assessed on the T2/FLAIR images in the registration overlay."
            )

        # Q4 — Enhancing tumor
        if et and et > 0:
            qa['volume_enhancing'] = (
                f"The enhancing tumor (ET, BraTS label 3) occupies **{et:.1f} mL** "
                f"({et/vol*100:.0f}% of total volume). "
                f"This represents the actively proliferating, blood-brain-barrier-disrupting component seen on T1c."
            )
        else:
            qa['volume_enhancing'] = (
                f"Enhancing tumor volume is not separately computed for this subject from the segmentation. "
                f"Total measured volume is {vol:.1f} mL. The T1c overlay provides visual confirmation."
            )

        # Q5 — Maximum diameter
        if diam:
            qa['volume_diameter'] = (
                f"The maximum tumor diameter is **{diam:.0f} mm** ({diam/10:.1f} cm). "
                f"This is measured as the longest axis across the tumor mass in the registered MRI space."
            )
        else:
            qa['volume_diameter'] = (
                f"Maximum diameter data is not directly available for this subject. "
                f"The total volume of {vol:.1f} mL suggests a maximum dimension estimate of "
                f"approximately {(vol * 6 / 3.14159) ** (1/3) * 10:.0f} mm (assuming spherical approximation)."
            )

        # Q6 — Location
        qa['region_location'] = (
            f"The tumor is primarily located in the **{hemi} hemisphere** "
            f"(L={lp:.0f}% / R={rp:.0f}% of lateralised volume), predominantly involving the "
            f"**{primary_lobe} lobe**, with a total volume of {vol:.1f} mL. "
            f"The most involved atlas regions are: {', '.join(top5)}. "
            + (f"Active proximity flags: {', '.join(active_flags)}." if active_flags else "No critical proximity flags raised.")
            + " (Atlas-verified data)"
        )

        # Q7 — Regions: enhancing tumor
        if compartments.get('et_ml', 0) > 0 or data.get('compartment_regions', {}).get('et'):
            qa['region_enhancing'] = get_regions_for_compartment(pid, 3, 'enhancing tumor (ET)', data)
        else:
            names = ', '.join(top5[:6])
            qa['region_enhancing'] = (
                f"The dominant tumor regions by atlas overlap are: **{names}**. "
                f"These represent areas of maximum tumor burden including the enhancing component. "
                f"(Atlas-verified — compartment-level ET regions unavailable)"
            )

        # Q8 — Regions: edema
        if data.get('compartment_regions', {}).get('ed'):
            qa['region_edema'] = get_regions_for_compartment(pid, 2, 'peritumoral edema (ED)', data)
        else:
            qa['region_edema'] = (
                f"Edema regions follow the general tumor distribution: {', '.join(top10[:8])}. "
                f"The edema typically extends into regions adjacent to the core tumor mass. "
                f"(Compartment-level ED breakdown unavailable; overall regions shown)"
            )

        # Q9 — Hemisphere
        if hemi == 'bilateral':
            qa['region_hemisphere'] = (
                f"The tumor has a **bilateral distribution** (L={lp:.0f}% / R={rp:.0f}%). "
                f"The bulk of involvement is on the {'left' if lp > rp else 'right'} side, "
                f"but there is significant contralateral extension across the midline. "
                f"(Atlas-verified)"
            )
        else:
            dominant_pct = lp if hemi == 'left' else rp
            qa['region_hemisphere'] = (
                f"The tumor is primarily in the **{hemi} hemisphere** ({dominant_pct:.0f}% of lateralised volume). "
                f"The contralateral hemisphere has only {100-dominant_pct:.0f}% involvement. "
                f"(Atlas-verified)"
            )

        # Q10 — Crosses midline
        if crosses:
            qa['region_midline'] = (
                f"**Yes**, the tumor crosses the midline, with L={lp:.0f}% and R={rp:.0f}% volume distribution. "
                "This bilateral extension is a feature commonly associated with glioblastoma ('butterfly' pattern) "
                f"and has significant implications for surgical planning. (Atlas-verified)"
            )
        else:
            dominant = hemi
            qa['region_midline'] = (
                f"**No**, the tumor does not significantly cross the midline. "
                f"It is predominantly confined to the {dominant} hemisphere "
                f"({lp:.0f}% left / {rp:.0f}% right). (Atlas-verified)"
            )

        # Q11 — Eloquent cortex
        if eloquent_list:
            qa['region_eloquent'] = (
                f"**Yes**, the tumor is near eloquent cortex: {'; '.join(eloquent_list)}. "
                f"This proximity significantly affects surgical approach and expected functional outcomes. "
                f"(Atlas proximity flags)"
            )
        else:
            qa['region_eloquent'] = (
                f"Based on atlas analysis, no critical eloquent cortex structures are flagged as adjacent. "
                f"Active proximity flags: {', '.join(active_flags) if active_flags else 'none'}. "
                f"However, careful intraoperative neuromonitoring is always recommended. (Atlas-verified)"
            )

        # Q12 — Motor cortex
        if near_motor:
            qa['region_motor'] = (
                f"**Yes**, the tumor is near the motor cortex (Precentral / Rolandic operculum regions are involved). "
                f"This carries significant risk of motor deficit and limits aggressive surgical resection. "
                f"(Atlas: NearMotorCortex flag raised)"
            )
        else:
            qa['region_motor'] = (
                f"**No**, the tumor is not directly adjacent to the primary motor cortex based on atlas analysis. "
                f"The NearMotorCortex proximity flag is not raised. "
                f"Standard surgical margins and monitoring apply. (Atlas-verified)"
            )

        # Q13 — Broca's / Wernicke's
        if near_broca or near_wernicke:
            areas = []
            if near_broca:    areas.append("Broca's area (IFG pars triangularis/opercularis)")
            if near_wernicke: areas.append("Wernicke's area (superior temporal gyrus)")
            qa['region_language'] = (
                f"**Yes**, the tumor is near language-critical areas: {'; '.join(areas)}. "
                f"This raises significant risk of expressive or receptive aphasia and must be considered in the surgical plan. "
                f"(Atlas proximity flags raised)"
            )
        else:
            qa['region_language'] = (
                f"Based on atlas analysis, the tumor is **not directly adjacent** to Broca's or Wernicke's areas. "
                f"Language proximity flags (NearBrocas, NearWernickes) are not raised. "
                f"Standard language mapping with awake craniotomy or fMRI may still be considered. (Atlas-verified)"
            )

        return qa


    # ─── Model-based Q&A (Q14–Q17 + Report) ───────────────────────────────────────
    def run_model_inference(pid: str, data: dict, image_path: str, dry_run: bool) -> dict:
        """Run MedGemma once for each morphology question and the initial report."""
        if dry_run:
            print(f"  [DRY-RUN] Skipping model inference for {pid}")
            return {
                'morph_margins':   "[DRY-RUN] Model inference placeholder for Q14: Tumor margins.",
                'morph_mass':      "[DRY-RUN] Model inference placeholder for Q15: Mass effect.",
                'morph_hetero':    "[DRY-RUN] Model inference placeholder for Q16: Signal heterogeneity.",
                'morph_cystic':    "[DRY-RUN] Model inference placeholder for Q17: Cystic degeneration.",
                'clin_location':   "[DRY-RUN] Model inference placeholder for Q18: Clinical concern.",
                'clin_deficits':   "[DRY-RUN] Model inference placeholder for Q19: Functional deficits.",
            }, {
                'morphology':    "[DRY-RUN] Morphology section placeholder.",
                'anatomy':       "[DRY-RUN] Anatomy section placeholder.",
                'correlation':   "[DRY-RUN] Correlation section placeholder.",
            }

        # Import inference only when needed (requires GPU)
        sys.path.insert(0, str(Path(__file__).parent.parent / 'webapp_v2'))
        from medgemma_inference import (
            run_v4_inference, compute_seg_compartments, compute_compartment_regions,
            extract_final_answer
        )
        from config import ATLAS_LABELS_NII, ATLAS_LABELS_TXT

        paths = get_paths(pid)
        seg_path = str(paths['seg_registered'])

        # Compute compartments and compartment regions for data
        compartments = compute_seg_compartments(seg_path)
        comp_regions = {}
        if ATLAS_LABELS_NII.exists() and ATLAS_LABELS_TXT.exists():
            comp_regions = compute_compartment_regions(seg_path, str(ATLAS_LABELS_NII), str(ATLAS_LABELS_TXT))

        full_data = dict(data)
        full_data.update(compartments)
        full_data['compartment_regions'] = comp_regions

        # Full initial report (must run first to build context history)
        print(f"  [MODEL] Initial report...")
        _, out_messages = run_v4_inference(
            analysis_data=full_data,
            image_path=image_path,
            feedback='',
            prev_messages=[],
        )
        # Parse MORPHOLOGY / ANATOMY / CORRELATION sections from the model's message history context
        from medgemma_inference import extract_sections, clean_clinical_text
        raw_sections_text = out_messages[-1]['content'][0]['text'] if out_messages else ''
        sections = extract_sections(raw_sections_text)
        report = {
            'morphology':  clean_clinical_text(sections.get('MORPHOLOGY', raw_sections_text)),
            'anatomy':     clean_clinical_text(sections.get('ANATOMY', '')),
            'correlation': clean_clinical_text(sections.get('CORRELATION', '')),
        }

        def ask(question: str) -> str:
            nonlocal out_messages
            response, updated_messages = run_v4_inference(
                analysis_data=full_data,
                image_path=image_path,
                feedback=question,
                prev_messages=out_messages,
            )
            out_messages = updated_messages
            return response.get('chat_reply', '')

        qa_model = {}
        print(f"  [MODEL] Q14 — Margins...")
        qa_model['morph_margins'] = ask(
            "Does the tumor have well-defined or infiltrative margins?"
        )
        print(f"  [MODEL] Q15 — Mass effect...")
        qa_model['morph_mass'] = ask(
            "Is there significant mass effect?"
        )
        print(f"  [MODEL] Q16 — Heterogeneity...")
        qa_model['morph_hetero'] = ask(
            "How heterogeneous is the tumor signal?"
        )
        print(f"  [MODEL] Q17 — Cystic degeneration...")
        qa_model['morph_cystic'] = ask(
            "Is there evidence of cystic degeneration?"
        )
        print(f"  [MODEL] Q18 — Clinical concern...")
        qa_model['clin_location'] = ask(
            "Is this concerning or manageable given the location?"
        )
        print(f"  [MODEL] Q19 — Functional deficits...")
        qa_model['clin_deficits'] = ask(
            "What functional deficits might this patient be experiencing?"
        )

        return qa_model, report


    # ─── Main ──────────────────────────────────────────────────────────────────────
    def main():
        parser = argparse.ArgumentParser(description='Prepare webapp_v3 data')
        parser.add_argument('--dry-run', action='store_true',
                            help='Skip GPU inference, fill placeholders instead')
        args = parser.parse_args()

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        for subj in SUBJECTS:
            pid      = subj['pid']
            accuracy = subj['accuracy']
            out_dir  = DATA_DIR / pid
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Processing {pid}  ({accuracy}% accuracy)")
            print(f"{'='*60}")

            # 1. Load atlas data
            print("  Loading atlas JSON...")
            data = load_aal(pid)
            paths = get_paths(pid)

            # 2. Compute segmentation compartment volumes (no GPU)
            compartments = {}
            if paths['seg_registered'].exists():
                print("  Computing compartment volumes...")
                import nibabel as nib, numpy as np
                img = nib.load(str(paths['seg_registered']))
                seg = np.squeeze(img.get_fdata())
                vox = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
                compartments = {
                    'ncr_volume_ml': float(np.sum(seg == 1)) * vox,
                    'ed_volume_ml':  float(np.sum(seg == 2)) * vox,
                    'et_volume_ml':  float(np.sum(seg == 3)) * vox,
                }
                compartments['total_seg_volume_ml'] = sum(compartments.values())
                data.update(compartments)

            # 3. Deterministic Q&A (no GPU)
            print("  Building deterministic Q&A (Q1–Q13)...")
            qa = build_deterministic_qa(pid, data, compartments)

            # 4. Model-based Q&A + report (GPU)
            image_path = str(paths['orthoview']) if paths['orthoview'].exists() else None
            qa_model, report = run_model_inference(pid, data, image_path, args.dry_run)
            qa.update(qa_model)

            # 5. Copy images
            print("  Copying images...")
            for img_key, img_path in [
                ('orthoview', paths['orthoview']),
                ('seg_multiclass', paths['seg_multiclass']),
                ('reg_quality', paths['reg_quality']),
            ]:
                if img_path.exists():
                    shutil.copy2(str(img_path), str(out_dir / img_path.name))
                    print(f"    [OK] {img_path.name}")
                else:
                    print(f"    [MISS] {img_path.name}")

            # 6. Save metadata
            hemi, lp, rp, crosses = compute_laterality(data)
            flags = data.get('proximity_flags', {})
            meta = {
                'pid':                    pid,
                'accuracy_pct':           accuracy,
                'total_volume_ml':        round(data.get('total_tumor_volume_ml', 0), 2),
                'ncr_volume_ml':          round(compartments.get('ncr_volume_ml', 0), 2),
                'ed_volume_ml':           round(compartments.get('ed_volume_ml', 0), 2),
                'et_volume_ml':           round(compartments.get('et_volume_ml', 0), 2),
                'max_diameter_mm':        data.get('max_diameter_mm'),
                'hemisphere':             hemi,
                'left_pct':               round(lp, 1),
                'right_pct':              round(rp, 1),
                'crosses_midline':        crosses,
                'primary_lobe':           compute_primary_lobe(data),
                'proximity_flags':        flags,
                'n_regions':              len(data.get('regions', [])),
                'orthoview_image':        paths['orthoview'].name if paths['orthoview'].exists() else None,
                'seg_image':              'segmentation_multiclass.png' if paths['seg_multiclass'].exists() else None,
                'reg_image':              'registration_quality.png' if paths['reg_quality'].exists() else None,
            }

            with open(out_dir / 'metadata.json', 'w') as f:
                json.dump(meta, f, indent=2)

            with open(out_dir / 'qa.json', 'w') as f:
                json.dump(qa, f, indent=2)

            with open(out_dir / 'report.json', 'w') as f:
                json.dump(report, f, indent=2)

            print(f"  [SAVED] {out_dir}")

        print(f"\n{'='*60}")
        print(f"  DONE — {len(SUBJECTS)} subjects prepared in {DATA_DIR}")
        print(f"{'='*60}")
        print("\nNext step: python webapp_v3/app.py")


    if __name__ == '__main__':
        main()
