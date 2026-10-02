//
//  ContentView.swift
//  Gilnun — redesigned UI (large text for low vision)
//

import SwiftUI

// MARK: - Palette
private enum P {
    static let bg          = Color(red:0.027, green:0.031, blue:0.055)
    static let primary     = Color(red:0.27,  green:0.72,  blue:1.0)
    static let live        = Color(red:0.16,  green:0.87,  blue:0.56)
    static let warning     = Color(red:1.0,   green:0.58,  blue:0.12)
    static let danger      = Color(red:1.0,   green:0.23,  blue:0.23)
    static let boxCaution  = Color(red:1.0,   green:0.85,  blue:0.0)   // yellow for caution boxes
    static let glass       = Color(red:0.09,  green:0.13,  blue:0.22).opacity(0.72)
    static let glassStroke = Color.white.opacity(0.10)
    static let dimText     = Color.white.opacity(0.45)
    static let pill        = Color.white.opacity(0.07)

    /// Risk-based box color: green when calm, yellow for caution, red for danger.
    /// Thresholds mirror `Severity` so the box color tracks the guidance card.
    static func riskColor(for score: Int) -> Color {
        if score >= 85 { return danger }      // red
        if score >= 55 { return boxCaution }  // yellow
        return live                           // green
    }
}

// MARK: - Enums
private enum AppMode: String, CaseIterable {
    case live = "실시간"
    case ocr  = "문자 읽기"
    var icon: String {
        self == .live ? "eye.fill" : "text.viewfinder"
    }
    var processingMode: CameraManager.ProcessingMode {
        self == .live ? .liveAnalyzing : .textDescription
    }
}

private enum Severity {
    case calm, warning, danger
    var color: Color {
        switch self {
        case .calm:    return P.primary
        case .warning: return P.warning
        case .danger:  return P.danger
        }
    }
    var label: String {
        switch self {
        case .calm:    return "안내"
        case .warning: return "주의"
        case .danger:  return "위험"
        }
    }
}

// MARK: - Root View
struct ContentView: View {
    @StateObject private var cam = CameraManager()
    @State private var mode: AppMode = .live

    var body: some View {
        ZStack {
            CameraPreview(session: cam.session)
                .ignoresSafeArea()

            VignetteLayer()

            if mode == .live {
                BoundingBoxOverlay(boxes: cam.liveBoxes, imageSize: cam.liveImageSize)
                    .ignoresSafeArea()
            }

            VStack(spacing: 0) {
                StatusBar()
                    .padding(.horizontal, 24)
                    .padding(.top, 14)

                ModePill(mode: mode)
                    .padding(.horizontal, 24)
                    .padding(.top, 16)

                Spacer()

                if mode == .live {
                    GuidanceCard(message: liveMessage, severity: severity)
                        .padding(.horizontal, 24)
                }

                if mode == .ocr {
                    OcrPanel(
                        status: cam.liveOCRStatus.rawValue,
                        result: cam.latestDetectedText
                    )
                    .padding(.horizontal, 24)
                }

                ModeBar(selected: $mode) { m in
                    cam.setMode(m.processingMode)
                }
                .padding(.horizontal, 18)
                .padding(.top, 12)
                .padding(.bottom, 30)
            }
        }
        .onAppear { cam.setMode(mode.processingMode) }
    }

    private var liveMessage: String { cam.latestGuide }

    private var severity: Severity {
        if cam.latestLiveRiskScore >= 85 { return .danger }
        if cam.latestLiveRiskScore >= 55 { return .warning }
        return .calm
    }
}

// MARK: - Camera Overlay
private struct VignetteLayer: View {
    var body: some View {
        LinearGradient(
            colors: [
                P.bg.opacity(0.82),
                P.bg.opacity(0.18),
                P.bg.opacity(0.92)
            ],
            startPoint: .top,
            endPoint: .bottom
        )
        .ignoresSafeArea()
    }
}

// MARK: - Bounding Box Overlay
/// Draws bounding boxes for only the most important (highest-risk) detections.
/// Decorative for sighted / low-vision demo clarity — hidden from VoiceOver so the
/// accessibility-first voice guidance remains the primary channel.
private struct BoundingBoxOverlay: View {
    let boxes: [LiveGuidanceBox]
    let imageSize: CGSize

    var body: some View {
        GeometryReader { geo in
            let bounds = CGRect(origin: .zero, size: geo.size)
            ZStack(alignment: .topLeading) {
                PreviewCenterMarker()
                    .position(x: geo.size.width / 2, y: geo.size.height / 2)

                ForEach(boxes) { box in
                    // Map into preview space, then clip to the visible preview bounds so a
                    // box never spills past the cropped edges of the aspect-fill image.
                    let frame = mapped(box.rect, view: geo.size).intersection(bounds)
                    if !frame.isNull, frame.width > 1, frame.height > 1 {
                        BoundingBoxView(
                            label: box.label,
                            color: P.riskColor(for: box.riskScore),
                            frame: frame
                        )
                    }
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
            .clipped()  // guarantees nothing (box edges or label) draws outside the preview
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    /// Maps a normalized (0...1) image-space rect onto the `.resizeAspectFill` camera
    /// preview. Aspect-fill scales the image by the *larger* axis ratio and center-crops
    /// the overflow, so we apply the same scale and centering offsets the preview layer
    /// uses — this is what keeps the box locked to the real object on screen.
    private func mapped(_ n: CGRect, view: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return .zero }
        let scale = max(view.width / imageSize.width, view.height / imageSize.height)
        let dispW = imageSize.width * scale
        let dispH = imageSize.height * scale
        let offX = (view.width - dispW) / 2
        let offY = (view.height - dispH) / 2
        return CGRect(
            x: offX + n.minX * dispW,
            y: offY + n.minY * dispH,
            width: n.width * dispW,
            height: n.height * dispH
        )
    }
}

private struct PreviewCenterMarker: View {
    var body: some View {
        ZStack {
            Rectangle()
                .fill(Color.white)
                .frame(width: 28, height: 2)
            Rectangle()
                .fill(Color.white)
                .frame(width: 2, height: 28)
            Circle()
                .stroke(Color.white, lineWidth: 2)
                .frame(width: 12, height: 12)
        }
        .shadow(color: .black.opacity(0.8), radius: 2, x: 0, y: 1)
    }
}

private struct BoundingBoxView: View {
    let label: String
    let color: Color
    let frame: CGRect

    var body: some View {
        // Lift the tag above the box top edge; tuck it inside when near the screen top.
        let tagAbove = frame.minY > 16

        ZStack(alignment: .topLeading) {
            Rectangle()
                .stroke(color, lineWidth: 2)
                .frame(width: max(0, frame.width), height: max(0, frame.height))

            Text(label)
                .font(.system(size: 11, weight: .heavy))
                .foregroundStyle(Color(red:0.02, green:0.06, blue:0.14))
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(color)
                .fixedSize()
                .offset(y: tagAbove ? -15 : 0)
        }
        .offset(x: frame.minX, y: frame.minY)
    }
}

private struct StatusBar: View {
    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 18, weight: .bold))
                .foregroundStyle(P.primary)

            Text("길눈")
                .font(.system(size: 18, weight: .heavy))
                .foregroundStyle(.white)

            Spacer()
        }
        .frame(height: 36)
    }
}

// MARK: - Mode Pill
private struct ModePill: View {
    let mode: AppMode

    var body: some View {
        HStack(spacing: 0) {
            Circle()
                .fill(mode == .live ? P.live : P.primary)
                .frame(width: 10, height: 10)  // 7→10
                .padding(.leading, 14)

            Text(mode == .live ? "보행 안내" : "문자 읽기")
                .font(.system(size: 17, weight: .bold))  // 13→17
                .foregroundStyle(.white)
                .padding(.leading, 9)

            Spacer()
        }
        .frame(height: 46)  // 36→46
        .background(P.pill, in: Capsule())
        .overlay(Capsule().stroke(P.glassStroke, lineWidth: 1))
    }
}

// MARK: - Guidance Card
private struct GuidanceCard: View {
    let message: String
    let severity: Severity

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Rectangle()
                    .fill(severity.color)
                    .frame(width: 5, height: 20)  // 3×14 → 5×20
                    .clipShape(.rect(cornerRadius: 3))
                Text(severity.label)
                    .font(.system(size: 14, weight: .bold))  // 10→14
                    .tracking(1.2)
                    .foregroundStyle(severity.color)
            }

            Text(message)
                .font(.system(size: 28, weight: .bold))  // 22→28
                .foregroundStyle(.white)
                .lineSpacing(5)
                .minimumScaleFactor(0.75)
                .lineLimit(3)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(22)  // 18→22
        .background(P.glass, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .stroke(severity.color.opacity(0.28), lineWidth: 1.5)
        )
    }
}

// MARK: - OCR Panel
private struct OcrPanel: View {
    let status: String
    let result: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(P.primary.opacity(0.14))
                    .frame(width: 36, height: 36)  // 30→36
                    .overlay(
                        Image(systemName: "text.viewfinder")
                            .font(.system(size: 17, weight: .semibold))  // 14→17
                            .foregroundStyle(P.primary)
                    )

                Text(status)
                    .font(.system(size: 17, weight: .semibold))  // 13→17
                    .foregroundStyle(P.dimText)

                Spacer()
            }

            if let text = result, !text.isEmpty {
                Divider()
                    .background(.white.opacity(0.08))
                    .padding(.vertical, 16)

                Text(text)
                    .font(.system(size: 32, weight: .bold))  // 26→32
                    .foregroundStyle(.white)
                    .lineSpacing(6)
            }
        }
        .padding(22)  // 18→22
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(P.glass, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .stroke(P.primary.opacity(0.25), lineWidth: 1.5)
        )
    }
}

// MARK: - Mode Bar
private struct ModeBar: View {
    @Binding var selected: AppMode
    let onChange: (AppMode) -> Void

    var body: some View {
        HStack(spacing: 8) {
            ForEach(AppMode.allCases, id: \.self) { m in
                ModeBtn(mode: m, active: selected == m) {
                    selected = m
                    onChange(m)
                }
            }
        }
        .padding(7)
        .background(P.glass, in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(P.glassStroke, lineWidth: 1)
        )
    }
}

private struct ModeBtn: View {
    let mode: AppMode
    let active: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: mode.icon)
                    .font(.system(size: 20, weight: .bold))  // 16→20
                Text(mode.rawValue)
                    .font(.system(size: 18, weight: .bold))  // 14→18
            }
            .foregroundStyle(active ? Color(red:0.02, green:0.06, blue:0.14) : .white.opacity(0.6))
            .frame(maxWidth: .infinity)
            .frame(height: 58)  // 48→58
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(active ? P.primary : P.pill)
                    .shadow(color: active ? P.primary.opacity(0.34) : .clear, radius: 14, x: 0, y: 7)
            )
        }
        .accessibilityLabel(mode.rawValue)
        .accessibilityAddTraits(active ? .isSelected : [])
    }
}
