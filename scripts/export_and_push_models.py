#!/usr/bin/env python3
"""
Export & ADB Push helper for Qualcomm AI Hub On-Device Image Generation models:
  - stable_diffusion_v1_5 (https://aihub.qualcomm.com/mobile/models/stable_diffusion_v1_5)
  - stable_diffusion_v2_1 (https://aihub.qualcomm.com/mobile/models/stable_diffusion_v2_1)
  - controlnet_canny      (https://aihub.qualcomm.com/mobile/models/controlnet_canny)

Usage:
  1. Install Qualcomm AI Hub Models:
     pip install "qai-hub-models[stable-diffusion-v1-5-quantized,stable-diffusion-v2-1-quantized,controlnet-canny-quantized]"
     qai-hub configure --api_token YOUR_API_TOKEN

  2. Export & push to connected Android device via ADB:
     python scripts/export_and_push_models.py --model all --device "Snapdragon 8 Elite QRD"

  3. Or push an already-exported local directory of .onnx files:
     python scripts/export_and_push_models.py --model stable_diffusion_v2_1 --skip-export --input-dir ./exported_sd21
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_PACKAGE = "com.example.on_device_image_generation"
DEVICE_BASE_DIR = f"/sdcard/Android/data/{APP_PACKAGE}/files/models"

MODEL_CONFIGS = {
    "stable_diffusion_v1_5": {
        "export_module": "qai_hub_models.models.stable_diffusion_v1_5_quantized.export",
        "components": {
            "text_encoder": "text_encoder.onnx",
            "unet": "unet.onnx",
            "vae_decoder": "vae_decoder.onnx",
        },
    },
    "stable_diffusion_v2_1": {
        "export_module": "qai_hub_models.models.stable_diffusion_v2_1_quantized.export",
        "components": {
            "text_encoder": "text_encoder.onnx",
            "unet": "unet.onnx",
            "vae_decoder": "vae_decoder.onnx",
        },
    },
    "controlnet_canny": {
        "export_module": "qai_hub_models.models.controlnet_canny_quantized.export",
        "components": {
            "text_encoder": "text_encoder.onnx",
            "controlnet": "controlnet.onnx",
            "unet": "unet.onnx",
            "vae_decoder": "vae_decoder.onnx",
        },
    },
}


def run_cmd(cmd: list[str]) -> None:
    print(f"-> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def export_model(model_key: str, output_dir: Path, device_name: str) -> None:
    cfg = MODEL_CONFIGS[model_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        cfg["export_module"],
        "--target-runtime",
        "onnx",
        "--device",
        device_name,
        "--output-dir",
        str(output_dir),
    ]
    run_cmd(cmd)


def push_model_to_device(model_key: str, source_dir: Path) -> None:
    cfg = MODEL_CONFIGS[model_key]
    target_dir = f"{DEVICE_BASE_DIR}/{model_key}"
    run_cmd(["adb", "shell", "mkdir", "-p", target_dir])

    onnx_files = list(source_dir.rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx files found in {source_dir}")

    for role_key, canonical_name in cfg["components"].items():
        matched = None
        for f in onnx_files:
            lower = f.name.lower().replace("-", "_")
            if role_key in lower or (role_key == "vae_decoder" and "vae" in lower):
                matched = f
                break
        if matched is None:
            print(f"[WARNING] Could not find component matching '{role_key}' in {source_dir}")
            continue

        print(f"Pushing {matched.name} -> {target_dir}/{canonical_name}")
        run_cmd(["adb", "push", str(matched), f"{target_dir}/{canonical_name}"])

        # Also push any external weight files (.onnx.data / .bin / .weights) in the same folder
        for sidecar in matched.parent.iterdir():
            if sidecar != matched and sidecar.is_file() and (
                sidecar.name.startswith(matched.stem) or sidecar.suffix in {".data", ".bin", ".weights"}
            ):
                run_cmd(["adb", "push", str(sidecar), f"{target_dir}/{sidecar.name}"])

    # Optional tokenizer files if present
    for tok_file in ("vocab.json", "merges.txt"):
        found = next(source_dir.rglob(tok_file), None)
        if found:
            run_cmd(["adb", "push", str(found), f"{target_dir}/{tok_file}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export & ADB Push Qualcomm AI Hub Diffusion models")
    parser.add_argument(
        "--model",
        choices=["all", *MODEL_CONFIGS.keys()],
        default="all",
        help="Which Qualcomm AI Hub model to export and push",
    )
    parser.add_argument(
        "--device",
        default="Snapdragon 8 Gen 3 QRD",
        help="Target Snapdragon device name for qai-hub export",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip running qai_hub_models export and only adb push existing files from --input-dir",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./exported_models"),
        help="Directory containing exported ONNX models",
    )
    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    for m in models:
        model_dir = args.input_dir / m if args.model == "all" else args.input_dir
        if not args.skip_export:
            export_model(m, model_dir, args.device)
        push_model_to_device(m, model_dir)

    print("\nAll requested models pushed! Tap 'Refresh' in the Android app to verify readiness.")


if __name__ == "__main__":
    main()

