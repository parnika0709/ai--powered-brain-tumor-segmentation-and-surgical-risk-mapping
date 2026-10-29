"""Service for running ML inference on MRI volumes."""

from __future__ import annotations

import time
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Any

from app.config import settings
from app.db import Prediction, get_session
from braintumor.registry import registry
from braintumor.schemas.contracts import VolumeInput, VolumeModality
from braintumor.inference.pipeline import run_pipeline
from braintumor.risk import compute_risk_regions, build_report_text

# Register the only predictor we ship, once, at import time.
# (The registry raises if you register the same name twice, so guard it.)
if "attn_unet_2d" not in registry.list_names():
    from braintumor.models.attention_unet_2d import build_attn_unet_2d

    registry.register("attn_unet_2d", build_attn_unet_2d)


class InferenceService:
    def __init__(self):
        self.predictor = registry.get(settings.active_predictor)
        weights_path = settings.model_weights_path or Path("runs/exp/best.pt")
        if weights_path.exists():
            self.predictor.load_weights(weights_path)

    def _load_volume_from_dir(self, case_dir: Path) -> VolumeInput:
        """
        Load modalities from a directory.
        Expects files with suffixes like _t1.nii.gz, _t1ce.nii.gz, etc.
        """
        ordered_modalities: list[VolumeModality] = []
        arrays = []
        spacing: tuple[float, float, float] | None = None
        affine: np.ndarray | None = None

        target_modalities = [VolumeModality.T1CE, VolumeModality.T2, VolumeModality.FLAIR]

        for mod in target_modalities:
            found = False
            for f in case_dir.iterdir():
                stem = f.name.replace(".nii.gz", "").replace(".nii", "")
                if stem.endswith(f"_{mod.value}"):
                    img = nib.load(str(f))
                    arr = img.get_fdata().astype(np.float32)
                    arrays.append(arr)
                    if spacing is None:
                        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
                    if affine is None:
                        affine = img.affine
                    ordered_modalities.append(mod)
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Missing required modality '{mod.value}' in {case_dir}. "
                    f"Expected a file ending in '_{mod.value}.nii' or '_{mod.value}.nii.gz'."
                )

        stacked = np.stack(arrays, axis=0)
        return VolumeInput(
            modalities=ordered_modalities,
            array=stacked,
            spacing=spacing,
            affine=affine,
        )

    def run_inference(self, case_id: int, case_dir: Path) -> Prediction:
        # 1. Load volume
        volume = self._load_volume_from_dir(case_dir)

        # 2. Run the full segmentation -> postprocess -> features pipeline.
        #    MC-Dropout uncertainty is skipped here to keep this basic/fast;
        #    flip compute_uncertainty=True if you want it later.
        pipeline_output = run_pipeline(
            self.predictor,
            volume,
            postprocess_mask=True,
            compute_uncertainty=False,
        )
        output = pipeline_output.prediction
        feature_bundle = pipeline_output.features

        # 3. Save mask (and uncertainty/saliency if present) to disk.
        case_results_dir = settings.results_dir / f"case_{case_id}"
        case_results_dir.mkdir(parents=True, exist_ok=True)

        mask_path = case_results_dir / "mask.nii.gz"
        nib.save(nib.Nifti1Image(output.mask.astype(np.uint8), volume.affine), str(mask_path))

        uncertainty_path = None
        if output.uncertainty is not None:
            uncertainty_path = case_results_dir / "uncertainty.nii.gz"
            nib.save(nib.Nifti1Image(output.uncertainty.astype(np.float32), volume.affine), str(uncertainty_path))

        saliency_path = None
        if output.saliency is not None:
            saliency_path = case_results_dir / "saliency.nii.gz"
            nib.save(nib.Nifti1Image(output.saliency.astype(np.float32), volume.affine), str(saliency_path))

        # 4. Heuristic surgical risk mapping (whole-tumor region vs. approximate
        #    functional regions — see braintumor.risk.mapping for the caveats).
        risk_regions = compute_risk_regions(
            tumor_mask=output.mask,
            volume_array=np.asarray(volume.array),
            spacing=volume.spacing,
        )
        wt_feature = next((f for f in feature_bundle.features if f.label.value == "whole_tumor"), None)
        tumor_volume_mm3 = wt_feature.volume_mm3 if wt_feature else None

        risk_report_text = build_report_text(
            case_name=f"case_{case_id}",
            regions=risk_regions,
            tumor_volume_mm3=tumor_volume_mm3,
        )
        risk_regions_json = [
            {
                "region": r.region,
                "function": r.function,
                "overlap_pct": r.overlap_pct,
                "distance_mm": r.distance_mm,
                "risk_tier": r.risk_tier,
            }
            for r in risk_regions
        ]

        # 5. Store in DB
        with get_session() as session:
            prediction = Prediction(
                case_id=case_id,
                model_name=self.predictor.name,
                model_version=self.predictor.version,
                inference_seconds=output.meta.inference_seconds,
                mask_path=str(mask_path),
                uncertainty_path=str(uncertainty_path) if uncertainty_path else None,
                saliency_path=str(saliency_path) if saliency_path else None,
                mesh_paths={},
                features=feature_bundle.model_dump(),
                metrics=None,
                risk_regions=risk_regions_json,
                risk_report=risk_report_text,
            )
            session.add(prediction)
            session.commit()
            session.refresh(prediction)
            return prediction
