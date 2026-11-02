package com.kojingapla.glass

enum class ProcessingMode {
    LIVE,
    TEXT
}

enum class LiveOcrStatus(val message: String) {
    SEARCHING("문자를 찾는 중..."),
    DETECTED("문자를 감지했습니다"),
    STABILIZING("화면을 안정화하는 중..."),
    READING("문자를 읽는 중..."),
    COOLING_DOWN("새 문자를 기다리는 중..."),
    UNAVAILABLE("카메라를 사용할 수 없습니다")
}

data class RectRatio(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
) {
    val width: Float get() = right - left
    val height: Float get() = bottom - top
    val centerX: Float get() = (left + right) / 2f
    val centerY: Float get() = (top + bottom) / 2f
}

/// A single bounding box to render over the live camera preview.
///
/// `rect` is normalized to the 0...1 image coordinate space so the view layer can
/// map it onto an aspect-fill (FILL_CENTER) camera preview at any screen size.
/// Mirrors iOS `LiveGuidanceBox`.
data class LiveGuidanceBox(
    val rect: RectRatio,
    val riskScore: Int,
    val label: String,
    /// "왼쪽", "정면" or "오른쪽".
    val positionLabel: String
)
