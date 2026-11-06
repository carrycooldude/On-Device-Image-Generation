package com.example.on_device_image_generation.model

import android.content.Context
import java.io.File

enum class QaiHubModelType(
    val id: String,
    val displayName: String,
    val subtitle: String,
    val hubUrl: String,
    val folderName: String,
    val qaiHubExportPackage: String,
    val textEmbedDim: Int,
    val maxTokenLength: Int = 77,
    val latentChannels: Int = 4,
    val latentSize: Int = 64,
    val imageSize: Int = 512,
    val requiresControlImage: Boolean = false
) {
    SD_V1_5(
        id = "stable_diffusion_v1_5",
        displayName = "Stable Diffusion v1.5",
        subtitle = "Text-to-Image • 512×512 • W8A16 QNN / ONNX",
        hubUrl = "https://aihub.qualcomm.com/mobile/models/stable_diffusion_v1_5",
        folderName = "stable_diffusion_v1_5",
        qaiHubExportPackage = "qai_hub_models.models.stable_diffusion_v1_5_quantized.export",
        textEmbedDim = 768,
        requiresControlImage = false
    ),
    SD_V2_1(
        id = "stable_diffusion_v2_1",
        displayName = "Stable Diffusion v2.1",
        subtitle = "High-Detail Text-to-Image • 512×512 • W8A16 QNN / ONNX",
        hubUrl = "https://aihub.qualcomm.com/mobile/models/stable_diffusion_v2_1",
        folderName = "stable_diffusion_v2_1",
        qaiHubExportPackage = "qai_hub_models.models.stable_diffusion_v2_1_quantized.export",
        textEmbedDim = 1024,
        requiresControlImage = false
    ),
    CONTROLNET_CANNY(
        id = "controlnet_canny",
        displayName = "ControlNet Canny",
        subtitle = "Canny Edge-Guided Generation • 512×512 • W8A16 QNN / ONNX",
        hubUrl = "https://aihub.qualcomm.com/mobile/models/controlnet_canny",
        folderName = "controlnet_canny",
        qaiHubExportPackage = "qai_hub_models.models.controlnet_canny_quantized.export",
        textEmbedDim = 768,
        requiresControlImage = true
    );

    fun requiredSubmodels(): List<SubmodelComponent> = buildList {
        add(
            SubmodelComponent(
                role = SubmodelRole.TEXT_ENCODER,
                primaryFileName = "text_encoder.onnx",
                candidateFileNames = listOf(
                    "text_encoder.onnx",
                    "TextEncoder.onnx",
                    "TextEncoder_Quantized.onnx",
                    "text_encoder_quantized.onnx"
                )
            )
        )
        if (requiresControlImage) {
            add(
                SubmodelComponent(
                    role = SubmodelRole.CONTROLNET,
                    primaryFileName = "controlnet.onnx",
                    candidateFileNames = listOf(
                        "controlnet.onnx",
                        "ControlNet.onnx",
                        "ControlNet_Quantized.onnx",
                        "controlnet_quantized.onnx"
                    )
                )
            )
        }
        add(
            SubmodelComponent(
                role = SubmodelRole.UNET,
                primaryFileName = "unet.onnx",
                candidateFileNames = listOf(
                    "unet.onnx",
                    "Unet.onnx",
                    "UNet.onnx",
                    "Unet_Quantized.onnx",
                    "unet_quantized.onnx"
                )
            )
        )
        add(
            SubmodelComponent(
                role = SubmodelRole.VAE_DECODER,
                primaryFileName = "vae_decoder.onnx",
                candidateFileNames = listOf(
                    "vae_decoder.onnx",
                    "VaeDecoder.onnx",
                    "VAE_Decoder.onnx",
                    "VaeDecoder_Quantized.onnx",
                    "vae_decoder_quantized.onnx"
                )
            )
        )
    }
}

enum class SubmodelRole(val label: String) {
    TEXT_ENCODER("Text Encoder"),
    CONTROLNET("ControlNet Canny"),
    UNET("UNet Denoiser"),
    VAE_DECODER("VAE Decoder")
}

data class SubmodelComponent(
    val role: SubmodelRole,
    val primaryFileName: String,
    val candidateFileNames: List<String>
)

enum class ExecutionProviderType(
    val displayName: String,
    val description: String
) {
    QNN_HTP(
        displayName = "Qualcomm QNN HTP (Hexagon NPU)",
        description = "Hardware-accelerated NPU execution via QNN Execution Provider (libQnnHtp.so)"
    ),
    QNN_CPU(
        displayName = "Qualcomm QNN CPU",
        description = "Qualcomm AI Engine Direct CPU backend (libQnnCpu.so)"
    ),
    NNAPI(
        displayName = "Android NNAPI",
        description = "Android Neural Networks API hardware delegate"
    ),
    CPU(
        displayName = "ONNX Runtime CPU",
        description = "Standard multi-threaded ARM64 XNNPACK / CPU execution"
    )
}

enum class SchedulerType(val displayName: String) {
    DPM_SOLVER_MULTISTEP("DPM-Solver++ (2M Karras)"),
    EULER_DISCRETE("Euler Discrete")
}

data class ComponentFileStatus(
    val component: SubmodelComponent,
    val resolvedFile: File?,
    val exists: Boolean,
    val sizeBytes: Long
)

data class ModelDirectoryStatus(
    val modelType: QaiHubModelType,
    val activeDirectory: File,
    val componentStatuses: List<ComponentFileStatus>,
    val vocabFile: File?,
    val mergesFile: File?
) {
    val isReady: Boolean
        get() = componentStatuses.all { it.exists && it.sizeBytes > 0L }

    val totalSizeMb: Double
        get() = componentStatuses.sumOf { it.sizeBytes } / (1024.0 * 1024.0)
}

class ModelRepository(private val context: Context) {

    private val prefs = context.getSharedPreferences("qai_hub_prefs", Context.MODE_PRIVATE)

    fun getCustomBaseDir(): String? = prefs.getString(KEY_CUSTOM_BASE_DIR, null)

    fun setCustomBaseDir(path: String?) {
        prefs.edit().apply {
            if (path.isNullOrBlank()) remove(KEY_CUSTOM_BASE_DIR)
            else putString(KEY_CUSTOM_BASE_DIR, path.trim())
        }.apply()
    }

    /**
     * Primary app-specific external storage directory that can be written via `adb push`
     * without requiring MANAGE_EXTERNAL_STORAGE permissions on Android 11+.
     */
    fun getDefaultBaseDir(): File {
        val extDir = context.getExternalFilesDir("models")
            ?: File(context.filesDir, "models")
        if (!extDir.exists()) {
            extDir.mkdirs()
        }
        return extDir
    }

    fun getCandidateModelDirs(modelType: QaiHubModelType): List<File> {
        val candidates = mutableListOf<File>()
        getCustomBaseDir()?.takeIf { it.isNotBlank() }?.let { customBase ->
            candidates.add(File(customBase, modelType.folderName))
            candidates.add(File(customBase))
        }
        val defaultBase = getDefaultBaseDir()
        candidates.add(File(defaultBase, modelType.folderName))
        candidates.add(File("/sdcard/Download/qai_hub_models/${modelType.folderName}"))
        candidates.add(File("/sdcard/qai_hub_models/${modelType.folderName}"))
        return candidates
    }

    fun ensureModelFoldersExist() {
        val base = getDefaultBaseDir()
        for (type in QaiHubModelType.entries) {
            val dir = File(base, type.folderName)
            if (!dir.exists()) {
                dir.mkdirs()
            }
        }
    }

    fun inspectModelStatus(modelType: QaiHubModelType): ModelDirectoryStatus {
        ensureModelFoldersExist()
        val candidateDirs = getCandidateModelDirs(modelType)

        // Pick the first directory that has at least one matching model file, or default to primary
        val chosenDir = candidateDirs.firstOrNull { dir ->
            dir.exists() && dir.isDirectory && modelType.requiredSubmodels().any { comp ->
                resolveComponentFile(dir, comp) != null
            }
        } ?: File(getDefaultBaseDir(), modelType.folderName)

        val statuses = modelType.requiredSubmodels().map { comp ->
            val file = resolveComponentFile(chosenDir, comp)
            ComponentFileStatus(
                component = comp,
                resolvedFile = file,
                exists = file != null && file.exists(),
                sizeBytes = file?.length() ?: 0L
            )
        }

        val vocabFile = findOptionalFile(chosenDir, getDefaultBaseDir(), "vocab.json")
        val mergesFile = findOptionalFile(chosenDir, getDefaultBaseDir(), "merges.txt")

        return ModelDirectoryStatus(
            modelType = modelType,
            activeDirectory = chosenDir,
            componentStatuses = statuses,
            vocabFile = vocabFile,
            mergesFile = mergesFile
        )
    }

    private fun resolveComponentFile(dir: File, component: SubmodelComponent): File? {
        if (!dir.exists() || !dir.isDirectory) return null
        for (candidate in component.candidateFileNames) {
            val exact = File(dir, candidate)
            if (exact.exists() && exact.isFile) return exact
        }
        // Also check case-insensitive prefix match for .onnx files in the folder
        val files = dir.listFiles() ?: return null
        val roleKeyword = when (component.role) {
            SubmodelRole.TEXT_ENCODER -> "text_encoder"
            SubmodelRole.CONTROLNET -> "controlnet"
            SubmodelRole.UNET -> "unet"
            SubmodelRole.VAE_DECODER -> "vae"
        }
        return files.firstOrNull { f ->
            f.isFile &&
                f.name.endsWith(".onnx", ignoreCase = true) &&
                f.name.lowercase().replace("-", "_").contains(roleKeyword)
        }
    }

    private fun findOptionalFile(modelDir: File, baseDir: File, fileName: String): File? {
        val inModelDir = File(modelDir, fileName)
        if (inModelDir.exists()) return inModelDir
        val inBaseDir = File(baseDir, fileName)
        if (inBaseDir.exists()) return inBaseDir
        return null
    }

    fun getAdbPushInstructions(modelType: QaiHubModelType): String {
        val targetPath = "/sdcard/Android/data/${context.packageName}/files/models/${modelType.folderName}"
        val components = modelType.requiredSubmodels().joinToString("\n") { comp ->
            "adb push ${comp.primaryFileName} $targetPath/${comp.primaryFileName}"
        }
        return buildString {
            appendLine("# 1. Export from Qualcomm AI Hub Models (ONNX Runtime target):")
            appendLine("python -m ${modelType.qaiHubExportPackage} --target-runtime onnx --output-dir ./exported_${modelType.folderName}")
            appendLine()
            appendLine("# 2. Push exported ONNX sub-models (and any .onnx.data weights) via ADB:")
            appendLine("adb shell mkdir -p $targetPath")
            appendLine(components)
            appendLine("# Optional CLIP vocabulary files:")
            appendLine("adb push vocab.json $targetPath/vocab.json")
            appendLine("adb push merges.txt $targetPath/merges.txt")
        }
    }

    companion object {
        private const val KEY_CUSTOM_BASE_DIR = "custom_base_dir"
    }
}

