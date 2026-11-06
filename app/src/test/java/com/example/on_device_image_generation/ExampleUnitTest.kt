package com.example.on_device_image_generation

import com.example.on_device_image_generation.model.QaiHubModelType
import com.example.on_device_image_generation.model.SchedulerType
import com.example.on_device_image_generation.model.SubmodelRole
import com.example.on_device_image_generation.scheduler.DiffusionScheduler
import com.example.on_device_image_generation.tokenizer.ClipTokenizer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ExampleUnitTest {

    @Test
    fun clipTokenizer_encodesWithBosEosAndFixed77Length() {
        val tokenizer = ClipTokenizer()
        val sd15Tokens = tokenizer.encode(
            "A futuristic cyberpunk city at sunset, ultra detailed",
            maxLength = 77,
            padWithZero = false
        )
        assertEquals(77, sd15Tokens.size)
        assertEquals(ClipTokenizer.BOS_TOKEN_ID, sd15Tokens.first())
        assertEquals(ClipTokenizer.EOS_TOKEN_ID, sd15Tokens.last())

        val sd21Tokens = tokenizer.encode(
            "A futuristic cyberpunk city at sunset, ultra detailed",
            maxLength = 77,
            padWithZero = true
        )
        assertEquals(77, sd21Tokens.size)
        assertEquals(ClipTokenizer.BOS_TOKEN_ID, sd21Tokens.first())
        assertEquals(0, sd21Tokens.last())
    }

    @Test
    fun dpmSolverScheduler_runsMultiStepDenoisingWithoutNaN() {
        val scheduler = DiffusionScheduler(SchedulerType.DPM_SOLVER_MULTISTEP)
        scheduler.setTimesteps(20)
        assertEquals(20, scheduler.timesteps.size)
        assertTrue(scheduler.timesteps.first() > scheduler.timesteps.last())

        var latents = scheduler.createInitialLatents(4 * 64 * 64, seed = 42L)
        for (stepIdx in scheduler.timesteps.indices) {
            val scaled = scheduler.scaleModelInput(latents, stepIdx)
            val mockNoise = FloatArray(scaled.size) { i -> scaled[i] * 0.1f }
            latents = scheduler.step(mockNoise, stepIdx, latents)
        }
        assertTrue(latents.all { it.isFinite() })
    }

    @Test
    fun eulerDiscreteScheduler_scalesInputAndUpdatesLatents() {
        val scheduler = DiffusionScheduler(SchedulerType.EULER_DISCRETE)
        scheduler.setTimesteps(15)
        assertEquals(15, scheduler.timesteps.size)
        assertTrue(scheduler.initNoiseSigma > 1.0f)

        var latents = scheduler.createInitialLatents(4 * 64 * 64, seed = 99L)
        for (stepIdx in scheduler.timesteps.indices) {
            val scaled = scheduler.scaleModelInput(latents, stepIdx)
            val mockNoise = FloatArray(scaled.size) { i -> scaled[i] * 0.05f }
            latents = scheduler.step(mockNoise, stepIdx, latents)
        }
        assertTrue(latents.all { it.isFinite() })
    }

    @Test
    fun sinusoidalTimeEmbedding_hasExpectedDimensionsAndBounds() {
        val emb = DiffusionScheduler.computeSinusoidalTimeEmbedding(timestep = 950, embeddingDim = 320)
        assertEquals(320, emb.size)
        assertTrue(emb.all { it in -1.0001f..1.0001f })
    }

    @Test
    fun qaiHubModelSpecs_matchExpectedSubmodelsAndDimensions() {
        assertEquals(768, QaiHubModelType.SD_V1_5.textEmbedDim)
        assertEquals(3, QaiHubModelType.SD_V1_5.requiredSubmodels().size)
        assertFalse(QaiHubModelType.SD_V1_5.requiresControlImage)

        assertEquals(1024, QaiHubModelType.SD_V2_1.textEmbedDim)
        assertEquals(3, QaiHubModelType.SD_V2_1.requiredSubmodels().size)
        assertFalse(QaiHubModelType.SD_V2_1.requiresControlImage)

        assertEquals(768, QaiHubModelType.CONTROLNET_CANNY.textEmbedDim)
        assertEquals(4, QaiHubModelType.CONTROLNET_CANNY.requiredSubmodels().size)
        assertTrue(QaiHubModelType.CONTROLNET_CANNY.requiresControlImage)
        assertTrue(
            QaiHubModelType.CONTROLNET_CANNY.requiredSubmodels()
                .any { it.role == SubmodelRole.CONTROLNET }
        )
    }
}
