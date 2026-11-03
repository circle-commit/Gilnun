package com.kojingapla.glass

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Port of iOS `TextAimingTests.swift`, with regions flipped to a top-left origin. */
class TextReadingTest {
    private fun rect(left: Float, top: Float, width: Float, height: Float) = RectRatio(left, top, left + width, top + height)

    @Test
    fun aimingHints() {
        val hint = TextStabilityTracker.Companion::aimingHint
        assertNull(hint(rect(0.3f, 0.4f, 0.4f, 0.2f), 0.0), "centered text needs no hint")
        assertEquals(TextAimingHint.MOVE_LEFT, hint(rect(0.0f, 0.4f, 0.5f, 0.2f), 0.0), "text cut on the left")
        assertEquals(TextAimingHint.MOVE_RIGHT, hint(rect(0.5f, 0.4f, 0.5f, 0.2f), 0.0), "text cut on the right")
        assertEquals(TextAimingHint.MOVE_UP, hint(rect(0.3f, 0.0f, 0.4f, 0.3f), 0.0), "text cut at the top")
        assertEquals(TextAimingHint.MOVE_DOWN, hint(rect(0.3f, 0.7f, 0.4f, 0.3f), 0.0), "text cut at the bottom")
        assertEquals(TextAimingHint.MOVE_FARTHER, hint(rect(0.0f, 0.4f, 1.0f, 0.2f), 0.0), "text wider than the frame")
        assertEquals(TextAimingHint.MOVE_CLOSER, hint(rect(0.45f, 0.45f, 0.1f, 0.1f), 0.0), "small text")
        assertEquals(TextAimingHint.MOVE_LEFT, hint(rect(0.05f, 0.4f, 0.3f, 0.2f), 0.0), "text off to the left")
        assertEquals(TextAimingHint.MOVE_UP, hint(rect(0.3f, 0.05f, 0.4f, 0.2f), 0.0), "text off to the top")
        assertEquals(TextAimingHint.GLARE, hint(rect(0.3f, 0.4f, 0.4f, 0.2f), 0.2), "glare on the text")
        assertEquals(TextAimingHint.MOVE_LEFT, hint(rect(0.0f, 0.4f, 0.5f, 0.2f), 0.2), "framing comes before glare")
    }

    @Test
    fun hintsAreSpacedAndTextIsReadAnyway() {
        val tracker = TextStabilityTracker()
        fun frame(seconds: Double, region: RectRatio) = tracker.update(
            OcrFrameAnalysis(region, 0.9f, 20.0, 0.0, 120.0, 0.0, (seconds * 1000).toLong(), listOf("약"))
        )
        val cut = rect(0.0f, 0.4f, 0.5f, 0.2f)
        assertNull(frame(0.0, cut).hint, "no hint on the first frame")
        assertNull(frame(0.3, cut).hint, "no hint before the delay")
        assertEquals(TextAimingHint.MOVE_LEFT, frame(0.6, cut).hint, "hint after the delay")
        assertNull(frame(0.9, cut).hint, "hint not repeated right away")
        assertEquals(TextAimingHint.MOVE_LEFT, frame(2.7, cut).hint, "hint repeated after the interval")
        assertTrue(frame(3.1, cut).shouldRunFullOcr, "cut-off text is read anyway after the fallback")

        val centered = rect(0.3f, 0.4f, 0.4f, 0.2f)
        tracker.reset()
        assertFalse(frame(10.0, centered).shouldRunFullOcr, "readable text waits to be steady")
        val steady = frame(10.8, centered)
        assertTrue(steady.shouldRunFullOcr && steady.hint == null, "steady readable text is read without a hint")
    }

    @Test
    fun duplicateText() {
        val suppressor = DuplicateTextSuppressor()
        val start = 1_000_000L
        assertTrue(suppressor.shouldSpeak("하루 세 번 식후 30분", start), "first text is read")
        assertFalse(suppressor.shouldSpeak("하루 세 번 식후 30분", start + 20_000), "the same text is not read again")
        assertFalse(suppressor.shouldSpeak("세 번 식후", start + 1_000), "a partial re-read stays quiet")
        assertTrue(suppressor.shouldSpeak("다음 페이지의 제목 세 가지", start + 2_000), "new text sharing one word is read")
    }

    @Test
    fun numberSafety() {
        val first = DetailRead(listOf("타이레놀정 500mg", "1일 3M 식후 30분", "식후에 드세요"), 0, 0)
        val second = DetailRead(listOf("타이레놀정 500 mg", "1일 3회 식후 30분", "식후에 드세요"), 0, 1)
        val confirmed = first.confirmedBy(second)
        assertEquals(listOf("타이레놀정 500mg", "식후에 드세요"), confirmed.lines, "only number lines read the same twice are kept")
        assertEquals(1, confirmed.droppedNumberLines, "the disagreeing number line is counted as dropped")
        val empty = first.confirmedBy(DetailRead.EMPTY)
        assertTrue(empty.lines == listOf("식후에 드세요") && empty.droppedNumberLines == 2, "no second read keeps no number lines")
    }

    @Test
    fun readingOrderAndConfidence() {
        fun line(text: String, left: Float, top: Float, confidence: Float = 0.9f) =
            OcrLine(text, RectRatio(left, top, left + 0.2f, top + 0.05f), confidence)
        val lines = listOf(
            line("둘째 줄", 0.1f, 0.5f),
            line("오른쪽", 0.5f, 0.31f),
            line("첫째 줄", 0.1f, 0.3f),
            line("흐린 글자", 0.1f, 0.7f, confidence = 0.4f),
            line("010-1234", 0.1f, 0.8f, confidence = 0.6f)
        )
        assertEquals(listOf("첫째 줄", "오른쪽", "둘째 줄", "010-1234"), OcrText.readableLines(lines), "rows top to bottom, left to right")
        val detail = OcrText.detailRead(lines)
        assertEquals(listOf("첫째 줄", "오른쪽", "둘째 줄"), detail.lines, "uncertain numbers are left out of a close read")
        assertEquals(1, detail.missedLines, "a line seen but not read is counted")
        assertEquals(1, detail.droppedNumberLines, "an uncertain number line is counted")
        assertEquals(RectRatio(0.1f, 0.3f, 0.7f, 0.85f), OcrText.textRegion(lines), "the region spans confident lines")
    }

    @Test
    fun glareFollowsRotation() {
        // Bright top rows of the sensor image are the right side of the upright frame at 90°.
        val values = DoubleArray(LumaSample.COLUMNS * LumaSample.ROWS) { index ->
            if (index / LumaSample.COLUMNS < LumaSample.ROWS / 2) 255.0 else 100.0
        }
        val sample = LumaSample(values)
        assertEquals(1.0, sample.glareRatio(RectRatio(0.7f, 0.3f, 0.9f, 0.7f), 90), "right side is glare at 90°")
        assertEquals(0.0, sample.glareRatio(RectRatio(0.1f, 0.3f, 0.3f, 0.7f), 90), "left side is clean at 90°")
        assertEquals(1.0, sample.glareRatio(RectRatio(0.3f, 0.1f, 0.7f, 0.3f), 0), "top is glare at 0°")
    }
}
