//
//  ContentView.swift
//  Gilnun — accessibility-first camera UI (Dynamic Type, Liquid Glass on iOS 26+)
//

import SwiftUI

// MARK: - Enums
private enum AppMode: String, CaseIterable {
    case live = "실시간"
    case ocr  = "문자 읽기"

    var icon: String {
        self == .live ? "eye.fill" : "text.viewfinder"
    }

    var accessibilityLabel: String {
        self == .live ? "실시간 보행 안내" : "문자 읽기"
    }

    var processingMode: CameraManager.ProcessingMode {
        self == .live ? .liveAnalyzing : .textDescription
    }
}

/// How urgent the current guidance is. Shown with color, an icon and a word, so it
/// never depends on color alone.
private enum Severity {
    case calm, warning, danger

    init(riskScore: Int) {
        if riskScore >= 85 {
            self = .danger
        } else if riskScore >= 55 {
            self = .warning
        } else {
            self = .calm
        }
    }

    var label: String {
        switch self {
        case .calm:    return "안내"
        case .warning: return "주의"
        case .danger:  return "위험"
        }
    }

    var icon: String {
        switch self {
        case .calm:    return "info.circle.fill"
        case .warning: return "exclamationmark.circle.fill"
        case .danger:  return "exclamationmark.triangle.fill"
        }
    }

    /// System colors, so "Increase Contrast" applies automatically.
    var color: Color {
        switch self {
        case .calm:    return .green
        case .warning: return .orange
        case .danger:  return .red
        }
    }
}

// MARK: - Root View
struct ContentView: View {
    @StateObject private var cam = CameraManager()
    @State private var mode: AppMode = .live
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            CameraPreview(session: cam.session)
                .ignoresSafeArea()
                .accessibilityHidden(true)

            TopScrim()
                .ignoresSafeArea()

            if mode == .live {
                BoundingBoxOverlay(boxes: cam.liveBoxes, imageSize: cam.liveImageSize)
                    .ignoresSafeArea()

                if severity == .danger {
                    DangerEdge()
                        .ignoresSafeArea()
                        .transition(.opacity)
                }
            }

            VStack(spacing: 12) {
                StatusChip(text: statusText, color: statusColor)

                Spacer(minLength: 0)

                if mode == .live, !cam.latestGuide.isEmpty {
                    GuidanceCard(message: cam.latestGuide, severity: severity)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }

                if mode == .ocr, let text = cam.latestDetectedText, !text.isEmpty {
                    OcrCard(text: text)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }

                ModeSwitcher(selected: $mode) { m in
                    cam.setMode(m.processingMode)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
        .preferredColorScheme(.dark)
        .animation(transition, value: severity)
        .animation(transition, value: cam.latestGuide)
        .animation(transition, value: mode)
        .onAppear { cam.setMode(mode.processingMode) }
    }

    private var transition: Animation? {
        reduceMotion ? nil : .smooth(duration: 0.3)
    }

    private var severity: Severity {
        Severity(riskScore: cam.latestLiveRiskScore)
    }

    private var statusText: String {
        if cam.isCameraDenied { return "카메라 권한이 필요해요" }
        switch mode {
        case .live: return cam.isLiveAnalysisRunning ? "주변을 살피는 중" : "카메라 준비 중"
        case .ocr:  return cam.liveOCRStatus.rawValue
        }
    }

    private var statusColor: Color {
        if cam.isCameraDenied { return .red }
        switch mode {
        case .live: return cam.isLiveAnalysisRunning ? .green : .gray
        case .ocr:  return .cyan
        }
    }
}

// MARK: - Surfaces
private extension View {
    /// Liquid Glass on iOS 26 and later; a frosted material with a hairline edge before that.
    /// Both follow the "Reduce Transparency" and "Increase Contrast" settings.
    @ViewBuilder
    func glassSurface(in shape: some Shape) -> some View {
        if #available(iOS 26.0, *) {
            glassEffect(.regular, in: shape)
        } else {
            background(.regularMaterial, in: shape)
                .overlay(shape.stroke(Color.white.opacity(0.15), lineWidth: 0.5))
        }
    }
}

// MARK: - Status Chip
private struct StatusChip: View {
    let text: String
    let color: Color

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(text)
                .font(.subheadline.weight(.semibold))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .glassSurface(in: Capsule())
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Guidance Card
private struct GuidanceCard: View {
    let message: String
    let severity: Severity

    /// Dark enough for white text to pass WCAG AA contrast.
    private static let dangerFill = Color(red: 0.70, green: 0.11, blue: 0.11)

    var body: some View {
        let parts = Self.split(message)
        let isDanger = severity == .danger

        VStack(alignment: .leading, spacing: 6) {
            Label(severity.label, systemImage: severity.icon)
                .font(.headline)
                .foregroundStyle(isDanger ? Color.white : severity.color)

            Text(parts.headline)
                .font(isDanger ? .largeTitle.bold() : .title.bold())
                .foregroundStyle(isDanger ? Color.white : Color.primary)

            if let detail = parts.detail {
                // Full-contrast text; size alone sets the hierarchy for low-vision readers.
                Text(detail)
                    .font(.title3)
                    .foregroundStyle(isDanger ? Color.white : Color.primary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(20)
        .modifier(CardBackground(severity: severity, dangerFill: Self.dangerFill))
        .accessibilityElement(children: .combine)
    }

    /// "잠시 멈춰 주세요. 정면에 차량이 가까워요." → headline + detail.
    static func split(_ message: String) -> (headline: String, detail: String?) {
        guard let end = message.range(of: ". ") else { return (message, nil) }

        let headline = String(message[..<end.lowerBound]) + "."
        let detail = message[end.upperBound...].trimmingCharacters(in: .whitespaces)
        return (headline, detail.isEmpty ? nil : detail)
    }
}

private struct CardBackground: ViewModifier {
    let severity: Severity
    let dangerFill: Color

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: 28, style: .continuous)

        switch severity {
        case .danger:
            content.background(dangerFill, in: shape)
        case .warning:
            content
                .glassSurface(in: shape)
                .overlay(shape.strokeBorder(severity.color, lineWidth: 2))
        case .calm:
            content.glassSurface(in: shape)
        }
    }
}

// MARK: - OCR Card
private struct OcrCard: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("읽은 문자", systemImage: "text.viewfinder")
                .font(.headline)
                .foregroundStyle(Color.cyan)

            Text(text)
                .font(.title2.bold())
                .lineLimit(8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(20)
        .glassSurface(in: RoundedRectangle(cornerRadius: 28, style: .continuous))
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Mode Switcher
private struct ModeSwitcher: View {
    @Binding var selected: AppMode
    let onChange: (AppMode) -> Void

    @Namespace private var selection
    @Environment(\.dynamicTypeSize) private var typeSize
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        HStack(spacing: 4) {
            ForEach(AppMode.allCases, id: \.self) { mode in
                Button {
                    withAnimation(reduceMotion ? nil : .snappy) {
                        selected = mode
                    }
                    onChange(mode)
                } label: {
                    segment(for: mode)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(mode.accessibilityLabel)
                .accessibilityAddTraits(selected == mode ? .isSelected : [])
            }
        }
        .padding(5)
        .glassSurface(in: Capsule())
    }

    private func segment(for mode: AppMode) -> some View {
        let isSelected = selected == mode
        // At accessibility text sizes the label goes under the icon so it still fits.
        let layout = typeSize.isAccessibilitySize
            ? AnyLayout(VStackLayout(spacing: 4))
            : AnyLayout(HStackLayout(spacing: 8))

        return layout {
            Image(systemName: mode.icon)
            Text(mode.rawValue)
        }
        .font(.headline)
        .foregroundStyle(isSelected ? Color.black : Color.primary)
        .frame(maxWidth: .infinity, minHeight: 52)
        .padding(.vertical, typeSize.isAccessibilitySize ? 8 : 0)
        .background {
            if isSelected {
                Capsule()
                    .fill(Color.white)
                    .matchedGeometryEffect(id: "selection", in: selection)
            }
        }
        .contentShape(Capsule())
    }
}

// MARK: - Top Scrim
/// Keeps the status bar legible over a bright camera image.
private struct TopScrim: View {
    var body: some View {
        VStack(spacing: 0) {
            LinearGradient(colors: [.black.opacity(0.5), .clear], startPoint: .top, endPoint: .bottom)
                .frame(height: 120)
            Spacer(minLength: 0)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

// MARK: - Danger Edge
/// A thick red frame around the whole screen while the guidance is "위험".
private struct DangerEdge: View {
    var body: some View {
        Rectangle()
            .strokeBorder(Color.red, lineWidth: 10)
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }
}

// MARK: - Bounding Box Overlay
/// Highlights the most important detection. Decorative for sighted and low-vision
/// users — hidden from VoiceOver so the voice guidance remains the primary channel.
private struct BoundingBoxOverlay: View {
    let boxes: [LiveGuidanceBox]
    let imageSize: CGSize

    var body: some View {
        GeometryReader { geo in
            let bounds = CGRect(origin: .zero, size: geo.size)
            ZStack(alignment: .topLeading) {
                ForEach(boxes) { box in
                    // Map into preview space, then clip to the visible preview bounds so a
                    // box never spills past the cropped edges of the aspect-fill image.
                    let frame = mapped(box.rect, view: geo.size).intersection(bounds)
                    if !frame.isNull, frame.width > 1, frame.height > 1 {
                        BoundingBoxView(box: box, frame: frame)
                    }
                }
            }
            // Pin the stack to the top-left; box offsets are measured from the preview origin.
            .frame(width: geo.size.width, height: geo.size.height, alignment: .topLeading)
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

private struct BoundingBoxView: View {
    let box: LiveGuidanceBox
    let frame: CGRect

    var body: some View {
        let color = Severity(riskScore: box.riskScore).color
        // Put the tag above the box; tuck it inside when the box reaches the screen top.
        let tagAbove = frame.minY > 44

        RoundedRectangle(cornerRadius: 14, style: .continuous)
            .strokeBorder(color, lineWidth: 4)
            .frame(width: frame.width, height: frame.height)
            .overlay(alignment: .topLeading) {
                Text("\(box.label) · \(box.positionLabel)")
                    .font(.footnote.bold())
                    .foregroundStyle(Color.black)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(color, in: Capsule())
                    .fixedSize()
                    .alignmentGuide(.top) { tagAbove ? $0[.bottom] + 6 : -8 }
                    .alignmentGuide(.leading) { _ in -6 }
            }
            .offset(x: frame.minX, y: frame.minY)
    }
}

// MARK: - Previews
#Preview("안내 카드") {
    VStack(spacing: 16) {
        GuidanceCard(message: "왼쪽에 벤치가 있어요.", severity: .calm)
        GuidanceCard(message: "정면에 기둥이 있어요. 천천히 이동해 주세요.", severity: .warning)
        GuidanceCard(message: "잠시 멈춰 주세요. 정면에 차량이 가까워요.", severity: .danger)
        OcrCard(text: "비상구는 왼쪽에 있습니다")
    }
    .padding()
    .background(Color.gray)
    .preferredColorScheme(.dark)
}

#Preview("모드 전환") {
    @Previewable @State var mode: AppMode = .live
    ModeSwitcher(selected: $mode) { _ in }
        .padding()
        .background(Color.gray)
        .preferredColorScheme(.dark)
}
