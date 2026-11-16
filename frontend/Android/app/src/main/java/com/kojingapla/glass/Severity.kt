package com.kojingapla.glass

/**
 * How urgent the current guidance is. Shown with color and a word, so it never depends on
 * color alone. Port of iOS `Severity` (ContentView.swift): the guidance card is filled with
 * the severity's color, and its text color keeps WCAG AA contrast over any camera image.
 */
enum class Severity(val label: String, val cardFill: Int, val ink: Int, val accent: Int) {
    CALM("안내", 0xFF0A4DB8.toInt(), 0xFFFFFFFF.toInt(), 0xFF45B8FF.toInt()),
    WARNING("주의", 0xFFFFB333.toInt(), 0xFF000000.toInt(), 0xFFFF941F.toInt()),
    DANGER("위험", 0xFFB31C1C.toInt(), 0xFFFFFFFF.toInt(), 0xFFFF3B3B.toInt());

    companion object {
        fun of(riskScore: Int): Severity = when {
            riskScore >= 85 -> DANGER
            riskScore >= 55 -> WARNING
            else -> CALM
        }
    }
}

object GuidanceText {
    /** "잠시 멈춰 주세요. 정면에 차량이 가까워요." → headline + detail, as iOS `GuidanceCard.split`. */
    fun split(message: String): Pair<String, String?> {
        val end = message.indexOf(". ")
        if (end < 0) return message to null
        val detail = message.substring(end + 2).trim()
        return message.substring(0, end) + "." to detail.ifEmpty { null }
    }
}
