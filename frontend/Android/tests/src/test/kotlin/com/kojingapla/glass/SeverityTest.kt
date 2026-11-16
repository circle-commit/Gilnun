package com.kojingapla.glass

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SeverityTest {
    @Test
    fun levelsMatchIos() {
        assertEquals(Severity.CALM, Severity.of(54))
        assertEquals(Severity.WARNING, Severity.of(55))
        assertEquals(Severity.WARNING, Severity.of(84), "a car beside the path (display risk 84) is 주의")
        assertEquals(Severity.DANGER, Severity.of(85))
        assertEquals(listOf("안내", "주의", "위험"), Severity.entries.map { it.label })
    }

    @Test
    fun cardTextKeepsContrast() {
        // WCAG AA asks 4.5:1 for normal text; the iOS colors were picked to pass it.
        for (severity in Severity.entries) {
            val ratio = contrast(severity.cardFill, severity.ink)
            assertTrue(ratio >= 4.5, "${severity.label}: contrast $ratio")
        }
    }

    @Test
    fun splitsHeadlineFromDetail() {
        assertEquals("잠시 멈춰 주세요." to "정면에 차량이 가까워요.", GuidanceText.split("잠시 멈춰 주세요. 정면에 차량이 가까워요."))
        assertEquals("왼쪽에 점자블록이 있어요." to null, GuidanceText.split("왼쪽에 점자블록이 있어요."))
        assertEquals("앞에 기둥이 있어요." to "천천히 이동해 주세요. 앞에 점자블록이 있어요.",
            GuidanceText.split("앞에 기둥이 있어요. 천천히 이동해 주세요. 앞에 점자블록이 있어요."))
    }

    private fun contrast(a: Int, b: Int): Double {
        fun luminance(color: Int): Double {
            fun channel(shift: Int): Double {
                val c = ((color shr shift) and 0xFF) / 255.0
                return if (c <= 0.03928) c / 12.92 else Math.pow((c + 0.055) / 1.055, 2.4)
            }
            return 0.2126 * channel(16) + 0.7152 * channel(8) + 0.0722 * channel(0)
        }
        val (light, dark) = listOf(luminance(a), luminance(b)).sortedDescending()
        return (light + 0.05) / (dark + 0.05)
    }
}
