package com.kojingapla.glass

class DuplicateTextSuppressor {
    private var lastNormalizedText = ""
    private var lastSpokenMs = Long.MIN_VALUE / 2
    private val cooldownMs = 8_000L

    fun shouldSpeak(text: String, nowMs: Long = System.currentTimeMillis()): Boolean {
        val normalized = normalize(text)
        if (normalized.isEmpty()) return false

        if (isDuplicate(normalized, lastNormalizedText)) return false

        // Shortly after speaking, a partial re-read of the same text stays quiet. New text that
        // only shares a common word (such as a page turn) is still read.
        if (nowMs - lastSpokenMs < cooldownMs) {
            val previousWords = lastNormalizedText.split(" ").filter { it.isNotEmpty() }.toSet()
            val newWords = normalized.split(" ").filter { it.isNotEmpty() }.toSet()
            val overlap = previousWords.intersect(newWords).size
            if (overlap >= minOf(previousWords.size, newWords.size) * 0.5) return false
        }

        lastNormalizedText = normalized
        lastSpokenMs = nowMs
        return true
    }

    private fun normalize(text: String): String =
        text.lowercase()
            .replace(Regex("[^\\p{IsAlphabetic}\\p{IsDigit}]+"), " ")
            .trim()
            .replace(Regex("\\s+"), " ")

    private fun isDuplicate(first: String, second: String): Boolean {
        if (first.isEmpty() || second.isEmpty()) return false
        if (first == second || first.contains(second) || second.contains(first)) return true

        val firstWords = first.split(" ").toSet()
        val secondWords = second.split(" ").toSet()
        val overlap = firstWords.intersect(secondWords).size
        val union = firstWords.union(secondWords).size
        return union > 0 && overlap.toDouble() / union.toDouble() >= 0.84
    }
}
