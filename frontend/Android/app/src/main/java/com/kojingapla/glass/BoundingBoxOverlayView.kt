package com.kojingapla.glass

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.view.View

/**
 * Draws bounding boxes for the highest-risk live detections, plus a center marker.
 *
 * Mirrors the iOS `BoundingBoxOverlay`: boxes are supplied in normalized (0...1)
 * image space and mapped onto the `FILL_CENTER` (aspect-fill) camera preview using
 * the same scale-by-larger-axis + center-crop math, so a box stays locked to the
 * real object on screen. Decorative only — hidden from the accessibility tree so the
 * spoken guidance remains the primary channel.
 */
class BoundingBoxOverlayView(context: Context) : View(context) {

    private var boxes: List<LiveGuidanceBox> = emptyList()
    private var braille: BrailleMask = BrailleMask.EMPTY
    private var imageW = 0f
    private var imageH = 0f

    private val density = resources.displayMetrics.density
    private fun dp(v: Float) = v * density

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
    }
    private val tagBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val tagTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = TAG_TEXT
        textSize = dp(11f)
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }
    /** Bright cyan like iOS: worn or gray blocks are hard to see. */
    private val braillePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = BRAILLE
        style = Paint.Style.FILL
    }
    private val markerFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }
    private val markerStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
    }

    init {
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
    }

    fun update(boxes: List<LiveGuidanceBox>, imageW: Float, imageH: Float, braille: BrailleMask = BrailleMask.EMPTY) {
        this.boxes = boxes
        this.braille = braille
        this.imageW = imageW
        this.imageH = imageH
        invalidate()
    }

    fun clear() {
        if (boxes.isEmpty() && braille.isEmpty) return
        boxes = emptyList()
        braille = BrailleMask.EMPTY
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val vw = width.toFloat()
        val vh = height.toFloat()
        canvas.save()
        canvas.clipRect(0f, 0f, vw, vh)

        // Center marker (white cross + ring), matching iOS PreviewCenterMarker.
        val cx = vw / 2f
        val cy = vh / 2f
        canvas.drawRect(cx - dp(14f), cy - dp(1f), cx + dp(14f), cy + dp(1f), markerFill)
        canvas.drawRect(cx - dp(1f), cy - dp(14f), cx + dp(1f), cy + dp(14f), markerFill)
        canvas.drawCircle(cx, cy, dp(6f), markerStroke)

        if (imageW <= 0f || imageH <= 0f || (boxes.isEmpty() && braille.isEmpty)) {
            canvas.restore()
            return
        }

        // Aspect-fill mapping: scale by the larger axis ratio and center-crop the overflow.
        val scale = maxOf(vw / imageW, vh / imageH)
        val dispW = imageW * scale
        val dispH = imageH * scale
        val offX = (vw - dispW) / 2f
        val offY = (vh - dispH) / 2f

        drawBraille(canvas, offX, offY, dispW, dispH)

        for (box in boxes) {
            val rawLeft = offX + box.rect.left * dispW
            val rawTop = offY + box.rect.top * dispH
            val rawRight = offX + box.rect.right * dispW
            val rawBottom = offY + box.rect.bottom * dispH

            // Clip to the visible preview so a box never spills past the cropped edges.
            val left = rawLeft.coerceIn(0f, vw)
            val top = rawTop.coerceIn(0f, vh)
            val right = rawRight.coerceIn(0f, vw)
            val bottom = rawBottom.coerceIn(0f, vh)
            if (right - left < 1f || bottom - top < 1f) continue

            val color = riskColor(box.riskScore)
            boxPaint.color = color
            canvas.drawRect(left, top, right, bottom, boxPaint)

            // Label tag: lifted above the box top edge, tucked inside when near the screen top.
            val label = box.label
            if (label.isEmpty()) continue
            val padH = dp(4f)
            val padV = dp(1f)
            val fm = tagTextPaint.fontMetrics
            val textH = fm.descent - fm.ascent
            val tagW = tagTextPaint.measureText(label) + padH * 2
            val tagH = textH + padV * 2
            val tagAbove = top > dp(16f)
            val unclampedTagBottom = if (tagAbove) top - dp(3f) else top + tagH
            val tagBottom = unclampedTagBottom.coerceIn(tagH, vh)
            val tagTop = (tagBottom - tagH).coerceIn(0f, vh - tagH)
            val tagLeft = left.coerceIn(0f, (vw - tagW).coerceAtLeast(0f))

            tagBgPaint.color = color
            canvas.drawRect(tagLeft, tagTop, tagLeft + tagW, tagTop + tagH, tagBgPaint)
            canvas.drawText(label, tagLeft + padH, tagTop + tagH - padV - fm.descent, tagTextPaint)
        }
        canvas.restore()
    }

    /** Paints the braille block cells, one rect per run of filled cells in a row. */
    private fun drawBraille(canvas: Canvas, offX: Float, offY: Float, dispW: Float, dispH: Float) {
        if (braille.isEmpty) return
        val cellW = dispW / BrailleMask.COLUMNS
        val cellH = dispH / BrailleMask.ROWS
        for (row in 0 until BrailleMask.ROWS) {
            var column = 0
            while (column < BrailleMask.COLUMNS) {
                if (!braille[column, row]) {
                    column += 1
                    continue
                }
                val start = column
                while (column < BrailleMask.COLUMNS && braille[column, row]) column += 1
                canvas.drawRect(offX + start * cellW, offY + row * cellH, offX + column * cellW, offY + (row + 1) * cellH, braillePaint)
            }
        }
    }

    /// Risk-based color: green when calm, yellow for caution, red for danger.
    private fun riskColor(score: Int): Int = when {
        score >= 85 -> DANGER
        score >= 55 -> CAUTION
        else -> LIVE
    }

    private companion object {
        const val DANGER = 0xFFFF3B3B.toInt()
        const val CAUTION = 0xFFFFD900.toInt()
        const val LIVE = 0xFF29DE8F.toInt()
        const val TAG_TEXT = 0xFF050F24.toInt()
        const val BRAILLE = 0x8000E6FF.toInt()
    }
}
