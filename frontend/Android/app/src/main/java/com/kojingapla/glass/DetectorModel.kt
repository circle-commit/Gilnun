package com.kojingapla.glass

/**
 * The two bundled detector files: the same YOLO11m, in float or quantized to int8
 * (`tools/quantize_onnx.py`). The user picks one in the settings.
 */
enum class DetectorModel(val asset: String, val title: String, val summary: String) {
    ACCURATE(
        "sidewalk_detector.onnx",
        "정확도 우선",
        "기본값. iOS와 같은 결과를 냅니다."
    ),
    FAST(
        "sidewalk_detector_int8.onnx",
        "속도 우선",
        "약 2배 빠릅니다. 멀리 있는 흐릿한 물체를 가끔 놓칠 수 있어요."
    );

    companion object {
        fun fromName(name: String?): DetectorModel = entries.firstOrNull { it.name == name } ?: ACCURATE
    }
}
