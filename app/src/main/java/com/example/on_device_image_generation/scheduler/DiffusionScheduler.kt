package com.example.on_device_image_generation.scheduler

import com.example.on_device_image_generation.model.SchedulerType
import java.util.Random
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Implements Stable Diffusion noise schedulers (DPM-Solver++ 2M Multistep & Euler Discrete)
 * matching Hugging Face `diffusers.DPMSolverMultistepScheduler` and Qualcomm AI Hub's
 * `qai_hub_models` Stable Diffusion v1.5, v2.1, and ControlNet Canny pipelines.
 */
class DiffusionScheduler(
    private val schedulerType: SchedulerType = SchedulerType.DPM_SOLVER_MULTISTEP,
    private val numTrainTimesteps: Int = 1000,
    betaStart: Float = 0.00085f,
    betaEnd: Float = 0.012f,
    private val predictionType: String = "epsilon"
) {
    private val betas: FloatArray = FloatArray(numTrainTimesteps)
    private val alphas: FloatArray = FloatArray(numTrainTimesteps)
    private val alphasCumprod: FloatArray = FloatArray(numTrainTimesteps)

    var timesteps: IntArray = intArrayOf()
        private set

    private var sigmas: FloatArray = floatArrayOf()
    private val modelOutputs = ArrayList<FloatArray>()

    val initNoiseSigma: Float
        get() = if (schedulerType == SchedulerType.EULER_DISCRETE && sigmas.isNotEmpty()) {
            sigmas[0]
        } else {
            1.0f
        }

    init {
        // Scaled linear beta schedule: linspace(sqrt(beta_start), sqrt(beta_end), 1000)^2
        val startSqrt = sqrt(betaStart.toDouble())
        val endSqrt = sqrt(betaEnd.toDouble())
        var cumprod = 1.0
        for (i in 0 until numTrainTimesteps) {
            val fraction = i.toDouble() / (numTrainTimesteps - 1).toDouble()
            val beta = (startSqrt + fraction * (endSqrt - startSqrt)).pow(2.0)
            val alpha = 1.0 - beta
            cumprod *= alpha
            betas[i] = beta.toFloat()
            alphas[i] = alpha.toFloat()
            alphasCumprod[i] = cumprod.toFloat()
        }
    }

    fun setTimesteps(numInferenceSteps: Int) {
        modelOutputs.clear()
        val steps = numInferenceSteps.coerceIn(1, numTrainTimesteps)
        // Matches Hugging Face: np.linspace(0, num_train_timesteps - 1, steps + 1).round()[::-1][:-1]
        timesteps = IntArray(steps) { idx ->
            val ratio = idx.toDouble() / steps.toDouble()
            ((numTrainTimesteps - 1) * (1.0 - ratio)).roundToInt().coerceIn(0, numTrainTimesteps - 1)
        }

        sigmas = FloatArray(steps + 1) { idx ->
            if (idx < steps) {
                val t = timesteps[idx]
                val acp = alphasCumprod[t].coerceIn(1e-6f, 0.999999f)
                sqrt((1.0f - acp) / acp)
            } else {
                0.0f
            }
        }
    }

    private fun sigmaToAlphaSigmaT(sigma: Float): Pair<Float, Float> {
        val alphaT = (1.0 / sqrt((sigma * sigma + 1.0f).toDouble())).toFloat()
        val sigmaT = sigma * alphaT
        return alphaT to sigmaT
    }

    /**
     * Scales the model input latent for schedulers that require sigma scaling (Euler).
     * For DPM-Solver++ in alpha-sigma space, returns a copy of sample unchanged.
     */
    fun scaleModelInput(sample: FloatArray, stepIndex: Int): FloatArray {
        if (schedulerType != SchedulerType.EULER_DISCRETE) {
            return sample.copyOf()
        }
        val sigma = sigmas[stepIndex]
        val scale = 1.0f / sqrt(sigma * sigma + 1.0f)
        return FloatArray(sample.size) { i -> sample[i] * scale }
    }

    /**
     * Performs a single reverse-diffusion denoising step.
     *
     * @param modelOutput The CFG-combined predicted noise (epsilon) of shape [1, 64, 64, 4].
     * @param stepIndex Index in `0 until timesteps.size`.
     * @param sample Current noisy latent tensor `x_t`.
     * @return Updated latent tensor `x_{t-1}`.
     */
    fun step(
        modelOutput: FloatArray,
        stepIndex: Int,
        sample: FloatArray
    ): FloatArray {
        return when (schedulerType) {
            SchedulerType.DPM_SOLVER_MULTISTEP -> stepDpmSolverMultistep(modelOutput, stepIndex, sample)
            SchedulerType.EULER_DISCRETE -> stepEulerDiscrete(modelOutput, stepIndex, sample)
        }
    }

    private fun stepDpmSolverMultistep(
        modelOutput: FloatArray,
        stepIndex: Int,
        sample: FloatArray
    ): FloatArray {
        val sigmaCur = sigmas[stepIndex]
        val sigmaNext = sigmas[stepIndex + 1]
        val (alphaS0, sigmaS0) = sigmaToAlphaSigmaT(sigmaCur)

        // Convert model output (epsilon or v_prediction) to x0_pred (data_prediction for DPM-Solver++)
        val x0Pred = if (predictionType == "v_prediction") {
            FloatArray(sample.size) { i ->
                alphaS0 * sample[i] - sigmaS0 * modelOutput[i]
            }
        } else {
            FloatArray(sample.size) { i ->
                (sample[i] - sigmaS0 * modelOutput[i]) / alphaS0
            }
        }

        modelOutputs.add(x0Pred)
        if (modelOutputs.size > 2) {
            modelOutputs.removeAt(0)
        }

        // Final step where sigma_t == 0.0: in HF DPMSolverMultistepScheduler (final_sigmas_type="zero"),
        // lower_order_final is true and (sigma_t / sigma_s) == 0, exp(-h) == 0, so x_t = x0_pred directly.
        if (stepIndex == timesteps.lastIndex || sigmaNext <= 0.0f) {
            return x0Pred
        }

        val (alphaT, sigmaT) = sigmaToAlphaSigmaT(sigmaNext)
        val lambdaT = ln(alphaT.toDouble()) - ln(sigmaT.toDouble())
        val lambdaS0 = ln(alphaS0.toDouble()) - ln(sigmaS0.toDouble())
        val h = (lambdaT - lambdaS0).toFloat()

        val sigmaRatio = sigmaT / sigmaS0
        val expMinusH = exp(-h.toDouble()).toFloat()
        val coeff = alphaT * (1.0f - expMinusH)

        val result = FloatArray(sample.size)
        if (modelOutputs.size == 1 || stepIndex == 0) {
            // First-order DPM-Solver++ update
            for (i in sample.indices) {
                result[i] = sigmaRatio * sample[i] + coeff * x0Pred[i]
            }
        } else {
            // Second-order multistep DPM-Solver++ (2M midpoint) update matching Hugging Face diffusers
            val sigmaPrev = sigmas[stepIndex - 1]
            val (alphaS1, sigmaS1) = sigmaToAlphaSigmaT(sigmaPrev)
            val lambdaS1 = ln(alphaS1.toDouble()) - ln(sigmaS1.toDouble())
            val h0 = (lambdaS0 - lambdaS1).toFloat()
            val r0 = h0 / h

            val m0 = modelOutputs[modelOutputs.size - 1]
            val m1 = modelOutputs[modelOutputs.size - 2]
            val halfInvR0 = 1.0f / (2.0f * r0)

            for (i in sample.indices) {
                val d0 = m0[i]
                val d1 = (1.0f / r0) * (m0[i] - m1[i])
                result[i] = sigmaRatio * sample[i] + coeff * d0 + 0.5f * coeff * d1
            }
        }
        return result
    }

    private fun stepEulerDiscrete(
        modelOutput: FloatArray,
        stepIndex: Int,
        sample: FloatArray
    ): FloatArray {
        val sigma = sigmas[stepIndex]
        val sigmaNext = sigmas[stepIndex + 1]
        val dt = sigmaNext - sigma

        return FloatArray(sample.size) { i ->
            sample[i] + modelOutput[i] * dt
        }
    }

    /**
     * Generates initial Gaussian latent noise of size `size` using the given seed.
     */
    fun createInitialLatents(size: Int, seed: Long): FloatArray {
        val rng = Random(seed)
        val scale = initNoiseSigma
        return FloatArray(size) {
            (rng.nextGaussian() * scale).toFloat()
        }
    }

    companion object {
        fun computeSinusoidalTimeEmbedding(timestep: Int, embeddingDim: Int = 320): FloatArray {
            val halfDim = embeddingDim / 2
            val maxPeriod = 10000.0
            val emb = FloatArray(embeddingDim)
            val logMaxPeriod = ln(maxPeriod) / halfDim.toDouble()
            for (i in 0 until halfDim) {
                val freq = exp(-logMaxPeriod * i.toDouble())
                val arg = timestep.toDouble() * freq
                emb[i] = cos(arg).toFloat()
                emb[i + halfDim] = sin(arg).toFloat()
            }
            return emb
        }
    }
}
