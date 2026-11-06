package com.example.on_device_image_generation.ui

import android.app.Application
import android.content.ContentValues
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.on_device_image_generation.inference.GenerationRequest
import com.example.on_device_image_generation.inference.GenerationResult
import com.example.on_device_image_generation.inference.OnnxQnnDiffusionPipeline
import com.example.on_device_image_generation.inference.StepProgress
import com.example.on_device_image_generation.model.ExecutionProviderType
import com.example.on_device_image_generation.model.ModelDirectoryStatus
import com.example.on_device_image_generation.model.ModelRepository
import com.example.on_device_image_generation.model.QaiHubModelType
import com.example.on_device_image_generation.model.SchedulerType
import com.example.on_device_image_generation.vision.CannyEdgeDetector
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ImageGenUiState(
    val selectedModel: QaiHubModelType = QaiHubModelType.SD_V2_1,
    val modelStatuses: Map<QaiHubModelType, ModelDirectoryStatus> = emptyMap(),
    val executionProvider: ExecutionProviderType = ExecutionProviderType.QNN_HTP,
    val schedulerType: SchedulerType = SchedulerType.DPM_SOLVER_MULTISTEP,
    val prompt: String = "A futuristic solarpunk citadel nestled in misty mountains at golden hour, cinematic lighting, ultra detailed, 8k",
    val negativePrompt: String = "ugly, blurry, low quality, poorly drawn, out of focus, distorted, disfigured, duplicate, watermark, text, signature, bad anatomy, low contrast, jpeg artifacts",
    val steps: Int = 20,
    val guidanceScale: Float = 8.0f,
    val seed: Long = 1337L,
    val controlSourceBitmap: Bitmap? = null,
    val cannyPreviewBitmap: Bitmap? = null,
    val cannyLowThreshold: Float = 80f,
    val cannyHighThreshold: Float = 180f,
    val controlNetScale: Float = 1.0f,
    val isGenerating: Boolean = false,
    val progress: StepProgress? = null,
    val lastResult: GenerationResult? = null,
    val errorMessage: String? = null,
    val statusMessage: String? = null,
    val customModelDir: String = ""
) {
    val activeModelStatus: ModelDirectoryStatus?
        get() = modelStatuses[selectedModel]
}

class ImageGenViewModel(application: Application) : AndroidViewModel(application) {

    val repository = ModelRepository(application.applicationContext)
    private val pipeline = OnnxQnnDiffusionPipeline(application.applicationContext)

    private val _uiState = MutableStateFlow(ImageGenUiState())
    val uiState: StateFlow<ImageGenUiState> = _uiState.asStateFlow()

    private var generationJob: Job? = null

    init {
        repository.ensureModelFoldersExist()
        val initialControl = CannyEdgeDetector.createSampleInputBitmap(
            CannyEdgeDetector.SamplePreset.CYBER_ARCHWAY
        )
        val initialEdges = CannyEdgeDetector.detectEdges(initialControl, 512, 80f, 180f)
        _uiState.update {
            it.copy(
                controlSourceBitmap = initialControl,
                cannyPreviewBitmap = initialEdges,
                customModelDir = repository.getCustomBaseDir().orEmpty()
            )
        }
        refreshModelStatuses()
    }

    fun refreshModelStatuses() {
        viewModelScope.launch(Dispatchers.IO) {
            val statuses = QaiHubModelType.entries.associateWith { type ->
                repository.inspectModelStatus(type)
            }
            _uiState.update {
                it.copy(
                    modelStatuses = statuses,
                    customModelDir = repository.getCustomBaseDir().orEmpty()
                )
            }
        }
    }

    fun selectModel(modelType: QaiHubModelType) {
        _uiState.update { it.copy(selectedModel = modelType, errorMessage = null) }
        refreshModelStatuses()
    }

    fun setExecutionProvider(provider: ExecutionProviderType) {
        _uiState.update { it.copy(executionProvider = provider) }
    }

    fun setScheduler(scheduler: SchedulerType) {
        _uiState.update { it.copy(schedulerType = scheduler) }
    }

    fun updatePrompt(prompt: String) {
        _uiState.update { it.copy(prompt = prompt) }
    }

    fun updateNegativePrompt(negativePrompt: String) {
        _uiState.update { it.copy(negativePrompt = negativePrompt) }
    }

    fun updateSteps(steps: Int) {
        _uiState.update { it.copy(steps = steps.coerceIn(1, 50)) }
    }

    fun updateGuidanceScale(scale: Float) {
        _uiState.update { it.copy(guidanceScale = scale.coerceIn(1.0f, 20.0f)) }
    }

    fun updateSeed(seed: Long) {
        _uiState.update { it.copy(seed = seed) }
    }

    fun randomizeSeed(): Long {
        val next = (1L..999_999_999L).random()
        _uiState.update { it.copy(seed = next) }
        return next
    }

    fun setControlImage(bitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.Default) {
            val resized = CannyEdgeDetector.centerCropResize(bitmap, 512, 512)
            val state = _uiState.value
            val edges = CannyEdgeDetector.detectEdges(
                source = resized,
                targetSize = 512,
                lowThreshold = state.cannyLowThreshold,
                highThreshold = state.cannyHighThreshold
            )
            _uiState.update {
                it.copy(
                    controlSourceBitmap = resized,
                    cannyPreviewBitmap = edges
                )
            }
        }
    }

    fun loadControlPreset(preset: CannyEdgeDetector.SamplePreset) {
        val sample = CannyEdgeDetector.createSampleInputBitmap(preset)
        setControlImage(sample)
    }

    fun updateCannyThresholds(low: Float, high: Float, controlScale: Float) {
        val safeLow = low.coerceIn(10f, 240f)
        val safeHigh = high.coerceIn(safeLow + 10f, 255f)
        _uiState.update {
            it.copy(
                cannyLowThreshold = safeLow,
                cannyHighThreshold = safeHigh,
                controlNetScale = controlScale.coerceIn(0.1f, 2.0f)
            )
        }
        val src = _uiState.value.controlSourceBitmap ?: return
        viewModelScope.launch(Dispatchers.Default) {
            val edges = CannyEdgeDetector.detectEdges(src, 512, safeLow, safeHigh)
            _uiState.update { it.copy(cannyPreviewBitmap = edges) }
        }
    }

    fun updateCustomModelDirectory(path: String?) {
        repository.setCustomBaseDir(path)
        refreshModelStatuses()
    }

    fun startGeneration() {
        if (_uiState.value.isGenerating) return

        val snapshot = _uiState.value
        val status = repository.inspectModelStatus(snapshot.selectedModel)
        _uiState.update {
            it.copy(
                modelStatuses = it.modelStatuses + (snapshot.selectedModel to status),
                errorMessage = null,
                statusMessage = null
            )
        }

        if (!status.isReady) {
            val missingList = status.componentStatuses
                .filter { !it.exists }
                .joinToString(", ") { it.component.primaryFileName }
            _uiState.update {
                it.copy(
                    errorMessage = "Model files not found for ${snapshot.selectedModel.displayName}.\n" +
                        "Missing: $missingList\n" +
                        "Folder: ${status.activeDirectory.absolutePath}\n" +
                        "Tap 'Model Setup / ADB Push' for instructions."
                )
            }
            return
        }

        val request = GenerationRequest(
            modelType = snapshot.selectedModel,
            prompt = snapshot.prompt.ifBlank { "A scenic landscape painting, highly detailed" },
            negativePrompt = snapshot.negativePrompt,
            numInferenceSteps = snapshot.steps,
            guidanceScale = snapshot.guidanceScale,
            seed = snapshot.seed,
            schedulerType = snapshot.schedulerType,
            executionProvider = snapshot.executionProvider,
            controlImageBitmap = snapshot.controlSourceBitmap,
            cannyLowThreshold = snapshot.cannyLowThreshold,
            cannyHighThreshold = snapshot.cannyHighThreshold,
            controlNetConditioningScale = snapshot.controlNetScale
        )

        generationJob = viewModelScope.launch(Dispatchers.Default) {
            _uiState.update {
                it.copy(
                    isGenerating = true,
                    errorMessage = null,
                    progress = StepProgress(
                        stage = "Preparing ${request.modelType.displayName}…",
                        currentStep = 0,
                        totalSteps = request.numInferenceSteps,
                        stepDurationMs = 0L,
                        activeProvider = request.executionProvider.displayName
                    )
                )
            }

            try {
                val result = pipeline.generate(status, request) { stepProgress ->
                    _uiState.update { state ->
                        state.copy(progress = stepProgress)
                    }
                }
                _uiState.update { state ->
                    state.copy(
                        isGenerating = false,
                        progress = null,
                        lastResult = result,
                        cannyPreviewBitmap = result.cannyEdgeBitmap ?: state.cannyPreviewBitmap,
                        statusMessage = "Generated in ${"%.1f".format(result.totalInferenceTimeMs / 1000.0)}s " +
                            "(${result.averageStepTimeMs} ms/step) on ${result.activeExecutionProvider}"
                    )
                }
            } catch (_: CancellationException) {
                _uiState.update {
                    it.copy(
                        isGenerating = false,
                        progress = null,
                        statusMessage = "Generation cancelled."
                    )
                }
            } catch (e: Throwable) {
                _uiState.update {
                    it.copy(
                        isGenerating = false,
                        progress = null,
                        errorMessage = "Inference error: ${e.message ?: e.javaClass.simpleName}"
                    )
                }
            }
        }
    }

    fun cancelGeneration() {
        generationJob?.cancel()
        generationJob = null
    }

    fun saveLastImageToGallery(onResult: (Uri?, String) -> Unit) {
        val bitmap = _uiState.value.lastResult?.outputBitmap ?: run {
            onResult(null, "No generated image to save.")
            return
        }
        val modelId = _uiState.value.selectedModel.id
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val resolver = getApplication<Application>().contentResolver
                val filename = "QAIHub_${modelId}_${System.currentTimeMillis()}.png"
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        put(
                            MediaStore.Images.Media.RELATIVE_PATH,
                            "${Environment.DIRECTORY_PICTURES}/QaiHubDiffusion"
                        )
                        put(MediaStore.Images.Media.IS_PENDING, 1)
                    }
                }
                val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    resolver.openOutputStream(uri)?.use { out ->
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                    }
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        values.clear()
                        values.put(MediaStore.Images.Media.IS_PENDING, 0)
                        resolver.update(uri, values, null, null)
                    }
                    withContext(Dispatchers.Main) {
                        onResult(uri, "Saved to Pictures/QaiHubDiffusion/$filename")
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        onResult(null, "Failed to create MediaStore entry.")
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    onResult(null, "Save failed: ${e.message}")
                }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        pipeline.closeSessions()
    }
}

