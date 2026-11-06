package com.example.on_device_image_generation.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.system.Os
import com.example.on_device_image_generation.model.ExecutionProviderType
import com.example.on_device_image_generation.model.ModelDirectoryStatus
import com.example.on_device_image_generation.model.QaiHubModelType
import com.example.on_device_image_generation.model.SchedulerType
import com.example.on_device_image_generation.model.SubmodelRole
import com.example.on_device_image_generation.scheduler.DiffusionScheduler
import com.example.on_device_image_generation.tokenizer.ClipTokenizer
import com.example.on_device_image_generation.vision.CannyEdgeDetector
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.IntBuffer
import kotlin.math.roundToInt

data class GenerationRequest(
    val modelType: QaiHubModelType,
    val prompt: String,
    val negativePrompt: String,
    val numInferenceSteps: Int = 20,
    val guidanceScale: Float = 7.5f,
    val seed: Long = 42L,
    val schedulerType: SchedulerType = SchedulerType.DPM_SOLVER_MULTISTEP,
    val executionProvider: ExecutionProviderType = ExecutionProviderType.QNN_HTP,
    val controlImageBitmap: Bitmap? = null,
    val cannyLowThreshold: Float = 80f,
    val cannyHighThreshold: Float = 180f,
    val controlNetConditioningScale: Float = 1.0f
)

data class StepProgress(
    val stage: String,
    val currentStep: Int,
    val totalSteps: Int,
    val stepDurationMs: Long,
    val activeProvider: String
)

data class GenerationResult(
    val outputBitmap: Bitmap,
    val cannyEdgeBitmap: Bitmap?,
    val activeExecutionProvider: String,
    val totalInferenceTimeMs: Long,
    val averageStepTimeMs: Long,
    val seedUsed: Long
)

/** I/O description of one tensor, read from the model bundle's metadata.json. */
private data class TensorSpec(
    val name: String,
    val shape: LongArray,
    val dtype: String,
    val scale: Float,
    val zeroPoint: Int
) {
    val isU16: Boolean get() = dtype == "uint16"
    val elementCount: Int get() = shape.fold(1L) { a, b -> a * b }.toInt()
}

/**
 * Runs Qualcomm AI Hub "precompiled_qnn_onnx" Stable Diffusion / ControlNet bundles
 * (small EPContext .onnx wrappers + .bin QNN context binaries) with the ONNX Runtime QNN EP.
 *
 * The bundles use NHWC layout and uint16 affine-quantized tensors, so every tensor is
 * quantized / dequantized with the scale + zero-point from metadata.json.
 */
class OnnxQnnDiffusionPipeline(private val context: Context) {

    private val ortEnv: OrtEnvironment =
        OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE, "ort-qnn")

    private var loadedModelType: QaiHubModelType? = null
    private var loadedProvider: ExecutionProviderType? = null
    private var providerLabel: String = "Uninitialized"

    private var textEncoder: OrtSession? = null
    private var controlNet: OrtSession? = null
    private var unet: OrtSession? = null
    private var vae: OrtSession? = null
    private var tokenizer: ClipTokenizer? = null

    // role key ("text_encoder", "unet", "vae", "controlnet") -> tensor specs
    private var inSpecs: Map<String, Map<String, TensorSpec>> = emptyMap()
    private var outSpecs: Map<String, Map<String, TensorSpec>> = emptyMap()
    private var socModel = 0
    private var htpArch = 0

    @Synchronized
    fun closeSessions() {
        runCatching { textEncoder?.close() }
        runCatching { controlNet?.close() }
        runCatching { unet?.close() }
        runCatching { vae?.close() }
        textEncoder = null; controlNet = null; unet = null; vae = null
        loadedModelType = null; loadedProvider = null
    }

    @Synchronized
    private fun ensureLoaded(
        status: ModelDirectoryStatus,
        provider: ExecutionProviderType,
        onStage: (String) -> Unit
    ) {
        check(status.isReady) {
            "Missing model files in ${status.activeDirectory.absolutePath}: " +
                status.componentStatuses.filter { !it.exists }.joinToString { it.component.primaryFileName }
        }
        if (loadedModelType == status.modelType && loadedProvider == provider &&
            textEncoder != null && unet != null && vae != null
        ) return

        closeSessions()
        configureNativeEnvironment()
        parseMetadata(File(status.activeDirectory, "metadata.json"))

        onStage("Initializing CLIP tokenizer…")
        tokenizer = ClipTokenizer(status.vocabFile, status.mergesFile)

        val files = status.componentStatuses.associate { it.component.role to it.resolvedFile!! }

        onStage("Loading text encoder on ${provider.displayName}…")
        textEncoder = createSession(files.getValue(SubmodelRole.TEXT_ENCODER), provider)
        if (status.modelType.requiresControlImage) {
            onStage("Loading ControlNet…")
            controlNet = createSession(files.getValue(SubmodelRole.CONTROLNET), provider)
        }
        onStage("Loading UNet…")
        unet = createSession(files.getValue(SubmodelRole.UNET), provider)
        onStage("Loading VAE decoder…")
        vae = createSession(files.getValue(SubmodelRole.VAE_DECODER), provider)

        providerLabel = provider.displayName
        loadedModelType = status.modelType
        loadedProvider = provider
    }

    suspend fun generate(
        status: ModelDirectoryStatus,
        request: GenerationRequest,
        onProgress: (StepProgress) -> Unit
    ): GenerationResult {
        val start = System.currentTimeMillis()
        fun report(stage: String, step: Int = 0, ms: Long = 0L) = onProgress(
            StepProgress(stage, step, request.numInferenceSteps, ms, providerLabel)
        )

        ensureLoaded(status, request.executionProvider) { report(it) }
        currentCoroutineContext().ensureActive()

        val tok = tokenizer!!
        val te = textEncoder!!
        val un = unet!!
        val vd = vae!!
        val cn = controlNet
        val modelType = status.modelType

        // 1. Text encoding (positive + negative prompt)
        report("Encoding prompt…")
        val padZero = modelType == QaiHubModelType.SD_V2_1
        val condEmb = runTextEncoder(te, tok.encode(request.prompt, padWithZero = padZero))
        val uncondEmb = runTextEncoder(te, tok.encode(request.negativePrompt, padWithZero = padZero))

        // 2. ControlNet conditioning image
        var cannyBitmap: Bitmap? = null
        var controlImage: FloatArray? = null
        if (cn != null) {
            report("Extracting Canny edges…")
            val src = request.controlImageBitmap
                ?: CannyEdgeDetector.createSampleInputBitmap(CannyEdgeDetector.SamplePreset.CYBER_ARCHWAY)
            cannyBitmap = CannyEdgeDetector.detectEdges(
                src, modelType.imageSize, request.cannyLowThreshold, request.cannyHighThreshold
            )
            controlImage = CannyEdgeDetector.bitmapToControlTensor(
                cannyBitmap, channelsFirst = false, targetSize = modelType.imageSize
            )
        }

        // 3. Scheduler + initial noise (latent layout is NHWC [1,64,64,4])
        val scheduler = DiffusionScheduler(schedulerType = request.schedulerType)
        scheduler.setTimesteps(request.numInferenceSteps)
        val latentCount = inSpecs.getValue("unet").getValue("latent").elementCount
        var latents = scheduler.createInitialLatents(latentCount, request.seed)

        val stepTimes = ArrayList<Long>()
        for (i in scheduler.timesteps.indices) {
            currentCoroutineContext().ensureActive()
            val t0 = System.currentTimeMillis()
            val timestep = scheduler.timesteps[i]
            val modelInput = scheduler.scaleModelInput(latents, i)

            val noiseUncond = runUnetPass(un, cn, modelInput, timestep, uncondEmb, controlImage, request.controlNetConditioningScale)
            val noiseCond = runUnetPass(un, cn, modelInput, timestep, condEmb, controlImage, request.controlNetConditioningScale)

            val guided = FloatArray(latentCount) { k ->
                noiseUncond[k] + request.guidanceScale * (noiseCond[k] - noiseUncond[k])
            }
            latents = scheduler.step(guided, i, latents)

            val dt = System.currentTimeMillis() - t0
            stepTimes.add(dt)
            report("Denoising step ${i + 1}/${scheduler.timesteps.size}", i + 1, dt)
        }

        currentCoroutineContext().ensureActive()
        report("Decoding image…", request.numInferenceSteps, stepTimes.lastOrNull() ?: 0L)
        val bitmap = runVae(vd, latents, modelType.imageSize)

        return GenerationResult(
            outputBitmap = bitmap,
            cannyEdgeBitmap = cannyBitmap,
            activeExecutionProvider = providerLabel,
            totalInferenceTimeMs = System.currentTimeMillis() - start,
            averageStepTimeMs = if (stepTimes.isEmpty()) 0L else stepTimes.average().toLong(),
            seedUsed = request.seed
        )
    }

    // ---------------------------------------------------------------- model runners

    private fun runTextEncoder(session: OrtSession, tokens: IntArray): FloatArray {
        val inSpec = inSpecs.getValue("text_encoder").values.first()
        val outSpec = outSpecs.getValue("text_encoder").values.first()
        OnnxTensor.createTensor(ortEnv, IntBuffer.wrap(tokens), inSpec.shape).use { input ->
            session.run(mapOf(inSpec.name to input)).use { result ->
                return readTensor(result.first { true }.value as OnnxTensor, outSpec)
            }
        }
    }

    private fun runUnetPass(
        unetSession: OrtSession,
        controlNetSession: OrtSession?,
        latent: FloatArray,
        timestep: Int,
        textEmb: FloatArray,
        controlImage: FloatArray?,
        controlScale: Float
    ): FloatArray {
        val unetIn = inSpecs.getValue("unet")
        val created = ArrayList<OnnxTensor>()
        try {
            val inputs = LinkedHashMap<String, OnnxTensor>()
            fun put(spec: TensorSpec, data: FloatArray) {
                val t = makeTensor(spec, data)
                created.add(t)
                inputs[spec.name] = t
            }
            put(unetIn.getValue("latent"), latent)
            put(unetIn.getValue("timestep"), floatArrayOf(timestep.toFloat()))
            put(unetIn.getValue("text_emb"), textEmb)

            if (controlNetSession != null && controlImage != null) {
                val cnIn = inSpecs.getValue("controlnet")
                val cnInputs = LinkedHashMap<String, OnnxTensor>()
                val cnCreated = ArrayList<OnnxTensor>()
                try {
                    fun cput(name: String, data: FloatArray) {
                        val spec = cnIn.getValue(name)
                        val t = makeTensor(spec, data)
                        cnCreated.add(t)
                        cnInputs[spec.name] = t
                    }
                    cput("text_emb", textEmb)
                    cput("image_cond", controlImage)
                    cput("latent", latent)
                    cput("timestep", floatArrayOf(timestep.toFloat()))

                    val cnOut = outSpecs.getValue("controlnet")
                    controlNetSession.run(cnInputs).use { res ->
                        for ((rawName, value) in res) {
                            val name = rawName.removeSuffix("_i16")
                            val outSpec = cnOut.getValue(name)
                            val unetName = when {
                                name == "mid_block" -> "controlnet_midblock"
                                name.startsWith("down_block_") -> "controlnet_downblock" + name.removePrefix("down_block_")
                                else -> continue
                            }
                            val data = readTensor(value as OnnxTensor, outSpec)
                            if (controlScale != 1.0f) for (k in data.indices) data[k] *= controlScale
                            put(unetIn.getValue(unetName), data)
                        }
                    }
                } finally {
                    cnCreated.forEach { runCatching { it.close() } }
                }
            }

            val outSpec = outSpecs.getValue("unet").values.first()
            unetSession.run(inputs).use { res ->
                return readTensor(res.first { true }.value as OnnxTensor, outSpec)
            }
        } finally {
            created.forEach { runCatching { it.close() } }
        }
    }

    private fun runVae(session: OrtSession, latents: FloatArray, size: Int): Bitmap {
        val inSpec = inSpecs.getValue("vae").values.first()
        val outSpec = outSpecs.getValue("vae").values.first()
        makeTensor(inSpec, latents).use { input ->
            session.run(mapOf(inSpec.name to input)).use { res ->
                val rgb = readTensor(res.first { true }.value as OnnxTensor, outSpec) // NHWC, [0,1]
                val pixels = IntArray(size * size)
                for (p in pixels.indices) {
                    val r = (rgb[p * 3].coerceIn(0f, 1f) * 255f + 0.5f).toInt()
                    val g = (rgb[p * 3 + 1].coerceIn(0f, 1f) * 255f + 0.5f).toInt()
                    val b = (rgb[p * 3 + 2].coerceIn(0f, 1f) * 255f + 0.5f).toInt()
                    pixels[p] = Color.rgb(r, g, b)
                }
                return Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
            }
        }
    }

    // ---------------------------------------------------------------- tensors / quantization

    private fun makeTensor(spec: TensorSpec, data: FloatArray): OnnxTensor {
        if (!spec.isU16) {
            return OnnxTensor.createTensor(ortEnv, java.nio.FloatBuffer.wrap(data), spec.shape)
        }
        // ORT Java has no UINT16 type: the wrapper graphs were patched to take INT16 and
        // Cast to UINT16 internally, so we pass the same 16 bits as a signed short.
        val buf = ByteBuffer.allocateDirect(data.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer()
        val inv = 1.0f / spec.scale
        for (x in data) {
            val q = ((x * inv).roundToInt() + spec.zeroPoint).coerceIn(0, 65535)
            buf.put(q.toShort())
        }
        buf.rewind()
        return OnnxTensor.createTensor(ortEnv, buf, spec.shape)
    }

    private fun readTensor(tensor: OnnxTensor, spec: TensorSpec): FloatArray {
        if (!spec.isU16) {
            val fb = tensor.floatBuffer
            return FloatArray(fb.remaining()).also { fb.get(it) }
        }
        val sb = tensor.shortBuffer
        val out = FloatArray(sb.remaining())
        for (i in out.indices) {
            out[i] = ((sb.get(i).toInt() and 0xFFFF) - spec.zeroPoint) * spec.scale
        }
        return out
    }

    // ---------------------------------------------------------------- session / metadata setup

    private fun configureNativeEnvironment() {
        // The Hexagon DSP loads its skel libraries from the filesystem, so point it at the
        // extracted native libs of this app.
        // ORT's QNN EP sets ADSP_LIBRARY_PATH itself from the backend_path directory.
        runCatching { Os.setenv("LD_LIBRARY_PATH", context.applicationInfo.nativeLibraryDir, true) }
    }

    private fun createSession(modelFile: File, provider: ExecutionProviderType): OrtSession {
        val nativeDir = context.applicationInfo.nativeLibraryDir
        val opts = OrtSession.SessionOptions()
        opts.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
        opts.setSessionLogVerbosityLevel(1)
        try {
            when (provider) {
                ExecutionProviderType.QNN_HTP -> opts.addQnn(
                    mapOf(
                        "backend_path" to backendPath(nativeDir, "libQnnHtp.so"),
                        "htp_performance_mode" to "burst"
                    )
                )
                ExecutionProviderType.QNN_CPU -> opts.addQnn(
                    mapOf("backend_path" to backendPath(nativeDir, "libQnnCpu.so"))
                )
                ExecutionProviderType.NNAPI -> opts.addNnapi()
                ExecutionProviderType.CPU -> Unit
            }
            return ortEnv.createSession(modelFile.absolutePath, opts)
        } catch (e: Exception) {
            throw IllegalStateException(
                "Failed to load ${modelFile.name} with ${provider.displayName}: ${e.message}", e
            )
        }
    }

    private fun backendPath(nativeDir: String?, lib: String): String {
        if (!nativeDir.isNullOrBlank()) {
            val f = File(nativeDir, lib)
            if (f.exists()) return f.absolutePath
        }
        return lib
    }

    private fun parseMetadata(file: File) {
        check(file.exists()) { "metadata.json not found in ${file.parent}" }
        val files = JSONObject(file.readText()).getJSONObject("model_files")
        JSONObject(file.readText()).optJSONObject("chipset_attributes")?.let {
            socModel = it.optInt("soc_model", 0)
            htpArch = it.optInt("htp_version", 0)
        }
        val ins = HashMap<String, Map<String, TensorSpec>>()
        val outs = HashMap<String, Map<String, TensorSpec>>()
        for (fileName in files.keys()) {
            val key = fileName.removeSuffix(".onnx")
            val obj = files.getJSONObject(fileName)
            ins[key] = parseSpecs(obj.getJSONObject("inputs"))
            outs[key] = parseSpecs(obj.getJSONObject("outputs"))
        }
        inSpecs = ins
        outSpecs = outs
    }

    private fun parseSpecs(obj: JSONObject): Map<String, TensorSpec> {
        val map = LinkedHashMap<String, TensorSpec>()
        for (name in obj.keys()) {
            val t = obj.getJSONObject(name)
            val shapeArr = t.getJSONArray("shape")
            val shape = LongArray(shapeArr.length()) { shapeArr.getLong(it) }
            val q = t.optJSONObject("quantization_parameters")
            val dtype = t.getString("dtype")
            val graphName = if (dtype == "uint16") "${name}_i16" else name
            map[name] = TensorSpec(
                name = graphName,
                shape = shape,
                dtype = dtype,
                scale = q?.getDouble("scale")?.toFloat() ?: 1f,
                zeroPoint = q?.getInt("zero_point") ?: 0
            )
        }
        return map
    }
}
