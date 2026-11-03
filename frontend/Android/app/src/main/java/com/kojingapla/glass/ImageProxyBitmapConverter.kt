package com.kojingapla.glass

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy

object ImageProxyBitmapConverter {
    /**
     * The frame turned upright, as the user holds the phone. With `cropToViewport` it is
     * limited to what the preview shows, so detection boxes line up with the screen.
     */
    fun toUprightBitmap(image: ImageProxy, cropToViewport: Boolean): Bitmap {
        val bitmap = image.toBitmap()
        val crop = image.cropRect
        val useCrop = cropToViewport && bitmap.width == image.width && bitmap.height == image.height &&
            !crop.isEmpty && (crop.width() != bitmap.width || crop.height() != bitmap.height)
        val left = if (useCrop) crop.left.coerceIn(0, bitmap.width - 1) else 0
        val top = if (useCrop) crop.top.coerceIn(0, bitmap.height - 1) else 0
        val width = if (useCrop) crop.width().coerceIn(1, bitmap.width - left) else bitmap.width
        val height = if (useCrop) crop.height().coerceIn(1, bitmap.height - top) else bitmap.height

        val rotation = image.imageInfo.rotationDegrees
        if (rotation == 0 && !useCrop) return bitmap
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(bitmap, left, top, width, height, matrix, true)
    }

    /** A coarse luma grid from the sensor-oriented Y plane (see `LumaSample`). */
    fun lumaSample(image: ImageProxy): LumaSample {
        val columns = LumaSample.COLUMNS
        val rows = LumaSample.ROWS
        val plane = image.planes.firstOrNull() ?: return LumaSample(DoubleArray(0))
        val buffer = plane.buffer.duplicate()
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val width = image.width
        val height = image.height
        val values = DoubleArray(columns * rows)

        for (row in 0 until rows) {
            val y = (row * height / rows).coerceIn(0, height - 1)
            for (column in 0 until columns) {
                val x = (column * width / columns).coerceIn(0, width - 1)
                values[row * columns + column] = (buffer.get(y * rowStride + x * pixelStride).toInt() and 0xFF).toDouble()
            }
        }
        return LumaSample(values)
    }
}
