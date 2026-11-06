package com.example.on_device_image_generation.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Shader
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * On-device Canny Edge Detector & tensor preprocessor for Qualcomm AI Hub's `controlnet_canny` model.
 */
object CannyEdgeDetector {

    enum class SamplePreset(val title: String) {
        CYBER_ARCHWAY("Archway & Skyline"),
        PORTRAIT_SILHOUETTE("Portrait Bust"),
        MOUNTAIN_SUNSET("Mountain & Sun")
    }

    /**
     * Resizes/center-crops [source] to [targetSize]x[targetSize] and runs Canny edge detection.
     */
    fun detectEdges(
        source: Bitmap,
        targetSize: Int = 512,
        lowThreshold: Float = 80f,
        highThreshold: Float = 180f
    ): Bitmap {
        val scaled = centerCropResize(source, targetSize, targetSize)
        val width = scaled.width
        val height = scaled.height
        val pixels = IntArray(width * height)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)

        // 1. Convert to grayscale [0..255]
        val gray = FloatArray(width * height)
        for (i in pixels.indices) {
            val c = pixels[i]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            gray[i] = 0.299f * r + 0.587f * g + 0.114f * b
        }

        // 2. 5x5 Gaussian blur (sigma ~ 1.4)
        val blurred = gaussianBlur5x5(gray, width, height)

        // 3. Sobel gradients (Gx, Gy) + magnitude + angle
        val mag = FloatArray(width * height)
        val dir = ByteArray(width * height) // 0: 0 deg, 1: 45 deg, 2: 90 deg, 3: 135 deg
        for (y in 1 until height - 1) {
            val row = y * width
            for (x in 1 until width - 1) {
                val idx = row + x
                val gx =
                    -blurred[(y - 1) * width + (x - 1)] + blurred[(y - 1) * width + (x + 1)] -
                        2f * blurred[y * width + (x - 1)] + 2f * blurred[y * width + (x + 1)] -
                        blurred[(y + 1) * width + (x - 1)] + blurred[(y + 1) * width + (x + 1)]

                val gy =
                    -blurred[(y - 1) * width + (x - 1)] - 2f * blurred[(y - 1) * width + x] - blurred[(y - 1) * width + (x + 1)] +
                        blurred[(y + 1) * width + (x - 1)] + 2f * blurred[(y + 1) * width + x] + blurred[(y + 1) * width + (x + 1)]

                mag[idx] = hypot(gx.toDouble(), gy.toDouble()).toFloat()

                var angle = Math.toDegrees(atan2(gy.toDouble(), gx.toDouble())).toFloat()
                if (angle < 0f) angle += 180f

                dir[idx] = when {
                    angle < 22.5f || angle >= 157.5f -> 0
                    angle < 67.5f -> 1
                    angle < 112.5f -> 2
                    else -> 3
                }
            }
        }

        // 4. Non-Maximum Suppression
        val nms = FloatArray(width * height)
        for (y in 1 until height - 1) {
            val row = y * width
            for (x in 1 until width - 1) {
                val idx = row + x
                val m = mag[idx]
                if (m <= 0f) continue
                val (n1, n2) = when (dir[idx].toInt()) {
                    0 -> mag[idx - 1] to mag[idx + 1]
                    1 -> mag[(y - 1) * width + (x + 1)] to mag[(y + 1) * width + (x - 1)]
                    2 -> mag[(y - 1) * width + x] to mag[(y + 1) * width + x]
                    else -> mag[(y - 1) * width + (x - 1)] to mag[(y + 1) * width + (x + 1)]
                }
                if (m >= n1 && m >= n2) {
                    nms[idx] = m
                }
            }
        }

        // 5. Double threshold & Hysteresis tracking
        val state = ByteArray(width * height) // 0 = non-edge, 1 = weak, 2 = strong
        val stack = IntArray(width * height)
        var stackTop = 0

        val low = lowThreshold.coerceAtLeast(1f)
        val high = highThreshold.coerceAtLeast(low + 1f)

        for (i in nms.indices) {
            val v = nms[i]
            if (v >= high) {
                state[i] = 2
                stack[stackTop++] = i
            } else if (v >= low) {
                state[i] = 1
            }
        }

        while (stackTop > 0) {
            val curr = stack[--stackTop]
            val cx = curr % width
            val cy = curr / width
            for (dy in -1..1) {
                val ny = cy + dy
                if (ny <= 0 || ny >= height - 1) continue
                for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = cx + dx
                    if (nx <= 0 || nx >= width - 1) continue
                    val nIdx = ny * width + nx
                    if (state[nIdx].toInt() == 1) {
                        state[nIdx] = 2
                        stack[stackTop++] = nIdx
                    }
                }
            }
        }

        val outPixels = IntArray(width * height)
        for (i in outPixels.indices) {
            outPixels[i] = if (state[i].toInt() == 2) Color.WHITE else Color.BLACK
        }

        return Bitmap.createBitmap(outPixels, width, height, Bitmap.Config.ARGB_8888)
    }

    /**
     * Converts a 512x512 Canny Edge Bitmap into a float array in `[0.0, 1.0]`
     * formatted either as NCHW `[1, 3, H, W]` or NHWC `[1, H, W, 3]`.
     */
    fun bitmapToControlTensor(
        edgeBitmap: Bitmap,
        channelsFirst: Boolean,
        targetSize: Int = 512
    ): FloatArray {
        val bmp = if (edgeBitmap.width == targetSize && edgeBitmap.height == targetSize) {
            edgeBitmap
        } else {
            Bitmap.createScaledBitmap(edgeBitmap, targetSize, targetSize, true)
        }
        val numPixels = targetSize * targetSize
        val pixels = IntArray(numPixels)
        bmp.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)

        val tensor = FloatArray(3 * numPixels)
        if (channelsFirst) {
            // NCHW: [1, 3, H, W]
            for (i in 0 until numPixels) {
                val c = pixels[i]
                val r = ((c shr 16) and 0xFF) / 255.0f
                val g = ((c shr 8) and 0xFF) / 255.0f
                val b = (c and 0xFF) / 255.0f
                tensor[i] = r
                tensor[numPixels + i] = g
                tensor[2 * numPixels + i] = b
            }
        } else {
            // NHWC: [1, H, W, 3]
            for (i in 0 until numPixels) {
                val c = pixels[i]
                val base = i * 3
                tensor[base] = ((c shr 16) and 0xFF) / 255.0f
                tensor[base + 1] = ((c shr 8) and 0xFF) / 255.0f
                tensor[base + 2] = (c and 0xFF) / 255.0f
            }
        }
        return tensor
    }

    fun centerCropResize(source: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val srcW = source.width
        val srcH = source.height
        val size = minOf(srcW, srcH)
        val x = (srcW - size) / 2
        val y = (srcH - size) / 2
        val cropped = Bitmap.createBitmap(source, x, y, size, size)
        return if (size == targetWidth && size == targetHeight) {
            cropped
        } else {
            Bitmap.createScaledBitmap(cropped, targetWidth, targetHeight, true)
        }
    }

    /**
     * Generates a synthetic test image for ControlNet Canny so users can test edge conditioning
     * immediately without needing to pick a gallery photo first.
     */
    fun createSampleInputBitmap(preset: SamplePreset, size: Int = 512): Bitmap {
        val bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)

        when (preset) {
            SamplePreset.CYBER_ARCHWAY -> {
                paint.shader = LinearGradient(
                    0f, 0f, 0f, size.toFloat(),
                    Color.rgb(20, 24, 48), Color.rgb(210, 120, 90),
                    Shader.TileMode.CLAMP
                )
                canvas.drawRect(0f, 0f, size.toFloat(), size.toFloat(), paint)
                paint.shader = null

                // Background skyline towers
                paint.color = Color.rgb(40, 48, 82)
                canvas.drawRect(40f, 160f, 115f, 430f, paint)
                canvas.drawRect(130f, 120f, 195f, 430f, paint)
                canvas.drawRect(315f, 130f, 385f, 430f, paint)
                canvas.drawRect(400f, 175f, 470f, 430f, paint)

                // Sun disc
                paint.color = Color.rgb(255, 225, 160)
                canvas.drawCircle(size / 2f, 220f, 65f, paint)

                // Foreground archway pillars + arch
                paint.color = Color.rgb(15, 18, 28)
                canvas.drawRect(0f, 420f, size.toFloat(), size.toFloat(), paint)
                canvas.drawRect(55f, 90f, 125f, 440f, paint)
                canvas.drawRect(size - 125f, 90f, size - 55f, 440f, paint)

                paint.style = Paint.Style.STROKE
                paint.strokeWidth = 42f
                canvas.drawArc(RectF(75f, 55f, size - 75f, 320f), 180f, 180f, false, paint)

                // Perspective pathway lines
                paint.strokeWidth = 5f
                paint.color = Color.rgb(240, 210, 170)
                canvas.drawLine(size / 2f - 30f, 420f, 110f, size.toFloat(), paint)
                canvas.drawLine(size / 2f + 30f, 420f, size - 110f, size.toFloat(), paint)
            }

            SamplePreset.PORTRAIT_SILHOUETTE -> {
                canvas.drawColor(Color.rgb(28, 32, 44))
                // Halo circle
                paint.color = Color.rgb(230, 190, 120)
                canvas.drawCircle(size / 2f, 200f, 135f, paint)

                // Shoulders & torso
                paint.color = Color.rgb(55, 65, 92)
                canvas.drawOval(RectF(95f, 310f, size - 95f, 560f), paint)

                // Neck & Head
                paint.color = Color.rgb(185, 145, 125)
                canvas.drawRect(size / 2f - 34f, 255f, size / 2f + 34f, 345f, paint)
                canvas.drawOval(RectF(size / 2f - 78f, 105f, size / 2f + 78f, 285f), paint)

                // Visor / glasses line for strong Canny structure
                paint.color = Color.rgb(25, 30, 45)
                canvas.drawRoundRect(RectF(size / 2f - 62f, 170f, size / 2f + 62f, 206f), 14f, 14f, paint)
            }

            SamplePreset.MOUNTAIN_SUNSET -> {
                canvas.drawColor(Color.rgb(35, 42, 72))
                // Large sun
                paint.color = Color.rgb(250, 200, 110)
                canvas.drawCircle(size * 0.62f, size * 0.34f, 74f, paint)

                // Far mountains
                val farPath = Path().apply {
                    moveTo(0f, 360f)
                    lineTo(130f, 210f)
                    lineTo(255f, 310f)
                    lineTo(380f, 175f)
                    lineTo(size.toFloat(), 330f)
                    lineTo(size.toFloat(), size.toFloat())
                    lineTo(0f, size.toFloat())
                    close()
                }
                paint.color = Color.rgb(85, 95, 135)
                canvas.drawPath(farPath, paint)

                // Near mountains
                val nearPath = Path().apply {
                    moveTo(0f, 400f)
                    lineTo(190f, 250f)
                    lineTo(310f, 365f)
                    lineTo(435f, 265f)
                    lineTo(size.toFloat(), 380f)
                    lineTo(size.toFloat(), size.toFloat())
                    lineTo(0f, size.toFloat())
                    close()
                }
                paint.color = Color.rgb(42, 50, 75)
                canvas.drawPath(nearPath, paint)

                // Lake reflection lines
                paint.color = Color.rgb(220, 195, 145)
                paint.strokeWidth = 4f
                for (row in 0..5) {
                    val y = 415f + row * 16f
                    val halfW = 70f - row * 8f
                    canvas.drawLine(size * 0.62f - halfW, y, size * 0.62f + halfW, y, paint)
                }
            }
        }
        return bmp
    }

    private fun gaussianBlur5x5(src: FloatArray, width: Int, height: Int): FloatArray {
        val kernel = intArrayOf(
            1, 4, 7, 4, 1,
            4, 16, 26, 16, 4,
            7, 26, 41, 26, 7,
            4, 16, 26, 16, 4,
            1, 4, 7, 4, 1
        )
        val out = FloatArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0f
                var weightSum = 0
                for (ky in -2..2) {
                    val ny = (y + ky).coerceIn(0, height - 1)
                    for (kx in -2..2) {
                        val nx = (x + kx).coerceIn(0, width - 1)
                        val w = kernel[(ky + 2) * 5 + (kx + 2)]
                        sum += src[ny * width + nx] * w
                        weightSum += w
                    }
                }
                out[y * width + x] = sum / weightSum
            }
        }
        return out
    }
}

