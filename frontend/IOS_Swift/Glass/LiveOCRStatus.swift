import SwiftUI

enum LiveOCRStatus: String {
    case searching = "문자를 찾는 중..."
    case detected = "문자를 감지했습니다"
    case stabilizing = "화면을 안정화하는 중..."
    case reading = "문자를 읽는 중..."
    case coolingDown = "새 문자를 기다리는 중..."
    case unavailable = "카메라를 사용할 수 없습니다"

    var accessibilityLabel: String {
        rawValue
    }
}

/// A single bounding box to render over the live camera preview.
///
/// `rect` is normalized to the 0...1 image coordinate space so the view layer can
/// map it onto an aspect-fill camera preview at any screen size.
struct LiveGuidanceBox: Identifiable {
    let id = UUID()
    let rect: CGRect
    let riskScore: Int
    let label: String
}
