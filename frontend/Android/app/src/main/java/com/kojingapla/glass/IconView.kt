package com.kojingapla.glass

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.view.View

/**
 * Lightweight Canvas-drawn glyphs standing in for the iOS SF Symbols used in the
 * Glass UI (`camera.viewfinder`, `eye.fill`, `text.viewfinder`). Avoids pulling in a
 * vector-asset / Material-icons dependency while keeping the look close to iOS.
 */
class IconView(context: Context, private val type: Type) : View(context) {

    enum class Type { CAMERA_VIEWFINDER, EYE, TEXT_VIEWFINDER }

    private var color = 0xFFFFFFFF.toInt()

    init {
        // Decorative: the button or label next to each icon carries the meaning for TalkBack.
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
    }

    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }

    fun setColor(c: Int) {
        if (c == color) return
        color = c
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        val s = minOf(w, h)
        if (s <= 0f) return

        stroke.color = color
        fill.color = color
        stroke.strokeWidth = s * 0.09f

        val cx = w / 2f
        val cy = h / 2f
        val pad = s * 0.14f

        when (type) {
            Type.CAMERA_VIEWFINDER -> {
                drawCorners(canvas, w, h, s, pad)
                canvas.drawCircle(cx, cy, s * 0.13f, stroke)
            }
            Type.TEXT_VIEWFINDER -> {
                drawCorners(canvas, w, h, s, pad)
                val lx0 = pad + s * 0.16f
                val lx1 = w - pad - s * 0.16f
                canvas.drawLine(lx0, cy - s * 0.10f, lx1, cy - s * 0.10f, stroke)
                canvas.drawLine(lx0, cy + s * 0.07f, lx0 + (lx1 - lx0) * 0.7f, cy + s * 0.07f, stroke)
            }
            Type.EYE -> {
                val ew = s * 0.42f
                val eh = s * 0.27f
                val path = Path().apply {
                    moveTo(cx - ew, cy)
                    quadTo(cx, cy - eh, cx + ew, cy)
                    quadTo(cx, cy + eh, cx - ew, cy)
                    close()
                }
                canvas.drawPath(path, stroke)
                canvas.drawCircle(cx, cy, s * 0.12f, fill)
            }
        }
    }

    /// Four viewfinder corner brackets inset by [pad].
    private fun drawCorners(canvas: Canvas, w: Float, h: Float, s: Float, pad: Float) {
        val len = s * 0.22f
        val l = pad
        val t = pad
        val r = w - pad
        val b = h - pad
        // top-left
        canvas.drawLine(l, t + len, l, t, stroke)
        canvas.drawLine(l, t, l + len, t, stroke)
        // top-right
        canvas.drawLine(r - len, t, r, t, stroke)
        canvas.drawLine(r, t, r, t + len, stroke)
        // bottom-left
        canvas.drawLine(l, b - len, l, b, stroke)
        canvas.drawLine(l, b, l + len, b, stroke)
        // bottom-right
        canvas.drawLine(r - len, b, r, b, stroke)
        canvas.drawLine(r, b - len, r, b, stroke)
    }
}
