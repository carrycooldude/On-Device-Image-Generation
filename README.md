# On-Device Image Generation (Qualcomm QNN HTP + ONNX Runtime)

Minimal Android app running **Qualcomm AI Hub** `precompiled_qnn_onnx` (`w8a16`) diffusion models directly on the **Snapdragon 8 Elite Hexagon NPU (QNN HTP)**:

- **Stable Diffusion v2.1** (`stable_diffusion_v2_1`) — 512×512 Text-to-Image (`OpenCLIP ViT-H/14`, 1024-dim)
- **Stable Diffusion v1.5** (`stable_diffusion_v1_5`) — 512×512 Text-to-Image (`CLIP ViT-L/14`, 768-dim)
- **ControlNet Canny** (`controlnet_canny`) — 512×512 Canny Edge-Guided Generation

## Features
- Hardware-accelerated on-device NPU inference via **ONNX Runtime Android QNN (`1.29.0`)** and **Qualcomm QNN Runtime (`2.50.0`)** (`libQnnHtp.so` + FastRPC `libcdsprpc.so`).
- Full `w8a16` (`uint16`) affine quantization/dequantization pipeline driven by each model bundle's `metadata.json`.
- Custom on-device **CLIP BPE Tokenizer** (`vocab.json` + `merges.txt`) and **DPM-Solver++ (2M Multistep)** / **Euler Discrete** schedulers.
- Clean UI with a model selector dropdown, prompt input, and `Generate` button.

## Model Setup (`adb push`)

1. Download the `precompiled_qnn_onnx` (`w8a16`) bundles from Qualcomm AI Hub for your target Snapdragon device:
   - [Stable Diffusion v2.1](https://aihub.qualcomm.com/mobile/models/stable_diffusion_v2_1)
   - [Stable Diffusion v1.5](https://aihub.qualcomm.com/mobile/models/stable_diffusion_v1_5)
   - [ControlNet Canny](https://aihub.qualcomm.com/mobile/models/controlnet_canny)
2. Patch the `.onnx` wrapper graphs for ONNX Runtime Java `INT16 <-> UINT16` compatibility:
   ```bash
   python scripts/patch_qnn_onnx_wrappers.py /path/to/bundle_dir
   ```
3. Push each model bundle (`.bin`, patched `.onnx`, `metadata.json`, plus CLIP `vocab.json` & `merges.txt`) to the app's external files directory:
   ```bash
   adb push <bundle_dir>/. /sdcard/Android/data/com.example.on_device_image_generation/files/models/<model_id>/
   ```
   where `<model_id>` is `stable_diffusion_v2_1`, `stable_diffusion_v1_5`, or `controlnet_canny`.

## Build

```bash
./gradlew assembleDebug
```
