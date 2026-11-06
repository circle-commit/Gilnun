package com.kojingapla.glass

import kotlin.math.floor
import kotlin.math.min

/**
 * Fits a frame into a model input the way Ultralytics' `LetterBox` does: scaled to fit with
 * OpenCV-style bilinear filtering (`INTER_LINEAR`), centered on gray (114) padding, written
 * as normalized planar RGB. Shared by the obstacle detector and the braille block
 * segmenter, like iOS `Letterbox`.
 */
class Letterbox(val frameWidth: Int, val frameHeight: Int, val modelWidth: Int, val modelHeight: Int) {
    val gain = min(modelWidth.toDouble() / frameWidth, modelHeight.toDouble() / frameHeight)
    private val scaledWidth = Math.rint(frameWidth * gain).toInt()
    private val scaledHeight = Math.rint(frameHeight * gain).toInt()
    private val left = Math.rint((modelWidth - scaledWidth) / 2.0 - 0.1).toInt()
    private val top = Math.rint((modelHeight - scaledHeight) / 2.0 - 0.1).toInt()

    /** Ultralytics `scale_boxes` padding: model input coordinates = frame coordinates * gain + pad. */
    val padX = Math.rint((modelWidth - frameWidth * gain) / 2 - 0.1)
    val padY = Math.rint((modelHeight - frameHeight * gain) / 2 - 0.1)

    /** Writes the ARGB `pixels` of the frame into `input` (3 x modelHeight x modelWidth). */
    fun render(pixels: IntArray, input: FloatArray) {
        val plane = modelWidth * modelHeight
        input.fill(PAD_VALUE)

        val scaleX = frameWidth.toDouble() / scaledWidth
        val scaleY = frameHeight.toDouble() / scaledHeight
        val sourceX0 = IntArray(scaledWidth)
        val sourceX1 = IntArray(scaledWidth)
        val weightX = FloatArray(scaledWidth)
        for (x in 0 until scaledWidth) {
            val position = (x + 0.5) * scaleX - 0.5
            val base = floor(position).toInt()
            weightX[x] = if (base < 0) 0f else (position - base).toFloat()
            sourceX0[x] = base.coerceIn(0, frameWidth - 1)
            sourceX1[x] = (base + 1).coerceIn(0, frameWidth - 1)
        }

        for (y in 0 until scaledHeight) {
            val targetY = top + y
            if (targetY !in 0 until modelHeight) continue
            val position = (y + 0.5) * scaleY - 0.5
            val base = floor(position).toInt()
            val weightY = if (base < 0) 0f else (position - base).toFloat()
            val row0 = base.coerceIn(0, frameHeight - 1) * frameWidth
            val row1 = (base + 1).coerceIn(0, frameHeight - 1) * frameWidth

            for (x in 0 until scaledWidth) {
                val targetX = left + x
                if (targetX !in 0 until modelWidth) continue
                val topLeft = pixels[row0 + sourceX0[x]]
                val topRight = pixels[row0 + sourceX1[x]]
                val bottomLeft = pixels[row1 + sourceX0[x]]
                val bottomRight = pixels[row1 + sourceX1[x]]
                val index = targetY * modelWidth + targetX
                for (channel in 0 until 3) {
                    val shift = 16 - channel * 8
                    val upper = lerp(channelOf(topLeft, shift), channelOf(topRight, shift), weightX[x])
                    val lower = lerp(channelOf(bottomLeft, shift), channelOf(bottomRight, shift), weightX[x])
                    input[channel * plane + index] = lerp(upper, lower, weightY) / 255f
                }
            }
        }
    }

    private fun channelOf(pixel: Int, shift: Int): Float = ((pixel shr shift) and 0xFF).toFloat()

    private fun lerp(start: Float, end: Float, weight: Float): Float = start + (end - start) * weight

    private companion object {
        const val PAD_VALUE = 114f / 255f
    }
}
