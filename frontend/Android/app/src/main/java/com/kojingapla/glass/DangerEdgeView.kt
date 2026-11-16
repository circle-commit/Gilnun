package com.kojingapla.glass

import android.content.Context
import android.graphics.Canvas
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Shader
import android.view.View

/**
 * A red glow spreading in from the screen's edges while the guidance is "위험". Port of
 * iOS `DangerEdge`: there a wide, blurred stroke on the rounded screen edge; here four
 * gradients from each edge inward, which look the same without a software-rendered blur.
 * Decorative: the spoken guidance and the card's "위험" carry the meaning.
 */
class DangerEdgeView(context: Context) : View(context) {
    private val depth = 90 * resources.displayMetrics.density
    private val paint = Paint()

    init {
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
        isClickable = false
    }

    override fun onDraw(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return
        edge(canvas, 0f, 0f, 0f, depth, 0f, 0f, w, depth)            // top
        edge(canvas, 0f, h, 0f, h - depth, 0f, h - depth, w, h)      // bottom
        edge(canvas, 0f, 0f, depth, 0f, 0f, 0f, depth, h)            // left
        edge(canvas, w, 0f, w - depth, 0f, w - depth, 0f, w, h)      // right
    }

    /** A gradient from (x0, y0) at full red to (x1, y1) transparent, filling the given rect. */
    private fun edge(
        canvas: Canvas,
        x0: Float, y0: Float, x1: Float, y1: Float,
        left: Float, top: Float, right: Float, bottom: Float
    ) {
        paint.shader = LinearGradient(x0, y0, x1, y1, intArrayOf(GLOW, GLOW_MID, 0), floatArrayOf(0f, 0.35f, 1f), Shader.TileMode.CLAMP)
        canvas.drawRect(left, top, right, bottom, paint)
    }

    private companion object {
        /** Red at 75% like iOS, fading out over the stroke's width. */
        const val GLOW = 0xBFFF2020.toInt()
        const val GLOW_MID = 0x66FF2020
    }
}
