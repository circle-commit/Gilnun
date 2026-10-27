import CoreGraphics
import Foundation

func runTextAimingTests() {
    // Vision coordinates: origin bottom-left.
    let hint = TextStabilityTracker.aimingHint
    expect(hint(CGRect(x: 0.3, y: 0.4, width: 0.4, height: 0.2), 0) == nil, "centered text needs no hint")
    expect(hint(CGRect(x: 0.0, y: 0.4, width: 0.5, height: 0.2), 0) == .moveLeft, "text cut on the left")
    expect(hint(CGRect(x: 0.5, y: 0.4, width: 0.5, height: 0.2), 0) == .moveRight, "text cut on the right")
    expect(hint(CGRect(x: 0.3, y: 0.7, width: 0.4, height: 0.3), 0) == .moveUp, "text cut at the top")
    expect(hint(CGRect(x: 0.3, y: 0.0, width: 0.4, height: 0.3), 0) == .moveDown, "text cut at the bottom")
    expect(hint(CGRect(x: 0.0, y: 0.4, width: 1.0, height: 0.2), 0) == .moveFarther, "text wider than the frame")
    expect(hint(CGRect(x: 0.45, y: 0.45, width: 0.1, height: 0.1), 0) == .moveCloser, "small text")
    expect(hint(CGRect(x: 0.05, y: 0.4, width: 0.3, height: 0.2), 0) == .moveLeft, "text off to the left")
    expect(hint(CGRect(x: 0.3, y: 0.75, width: 0.4, height: 0.2), 0) == .moveUp, "text off to the top")
    expect(hint(CGRect(x: 0.3, y: 0.4, width: 0.4, height: 0.2), 0.2) == .glare, "glare on the text")
    expect(hint(CGRect(x: 0.0, y: 0.4, width: 0.5, height: 0.2), 0.2) == .moveLeft, "framing comes before glare")

    // Spoken only after the problem lasts, then spaced out, and read anyway after a while.
    let tracker = TextStabilityTracker()
    let start = Date()
    func frame(_ seconds: Double, _ region: CGRect) -> TextStabilityDecision {
        tracker.update(with: OCRFrameAnalysis(
            textRegion: region, confidence: 0.9, blurScore: 20, movementScore: 0, brightness: 120, glareRatio: 0,
            timestamp: start.addingTimeInterval(seconds), readableLines: ["약"]
        ))
    }
    let cut = CGRect(x: 0.0, y: 0.4, width: 0.5, height: 0.2)
    expect(frame(0.0, cut).hint == nil, "no hint on the first frame")
    expect(frame(0.3, cut).hint == nil, "no hint before the delay")
    expect(frame(0.6, cut).hint == .moveLeft, "hint after the delay")
    expect(frame(0.9, cut).hint == nil, "hint not repeated right away")
    expect(frame(2.7, cut).hint == .moveLeft, "hint repeated after the interval")
    expect(frame(3.1, cut).shouldRunFullOCR, "cut-off text is read anyway after the fallback")

    let centered = CGRect(x: 0.3, y: 0.4, width: 0.4, height: 0.2)
    tracker.reset()
    expect(!frame(10.0, centered).shouldRunFullOCR, "readable text waits to be steady")
    let steady = frame(10.8, centered)
    expect(steady.shouldRunFullOCR && steady.hint == nil, "steady readable text is read without a hint")
}

func runDuplicateTextTests() {
    let suppressor = DuplicateTextSuppressor()
    let start = Date()
    expect(suppressor.shouldSpeak("하루 세 번 식후 30분", now: start), "first text is read")
    expect(!suppressor.shouldSpeak("하루 세 번 식후 30분", now: start.addingTimeInterval(20)), "the same text is not read again")
    expect(!suppressor.shouldSpeak("세 번 식후", now: start.addingTimeInterval(1)), "a partial re-read stays quiet")
    expect(suppressor.shouldSpeak("다음 페이지의 제목 세 가지", now: start.addingTimeInterval(2)), "new text sharing one word is read")
}

func runNumberSafetyTests() {
    let first = DetailRead(lines: ["타이레놀정 500mg", "1일 3M 식후 30분", "식후에 드세요"], missedLines: 0, droppedNumberLines: 0)
    let second = DetailRead(lines: ["타이레놀정 500 mg", "1일 3회 식후 30분", "식후에 드세요"], missedLines: 0, droppedNumberLines: 1)
    let confirmed = first.confirmed(by: second)
    expect(confirmed.lines == ["타이레놀정 500mg", "식후에 드세요"], "only number lines read the same twice are kept: \(confirmed.lines)")
    expect(confirmed.droppedNumberLines == 1, "the disagreeing number line is counted as dropped")
    let empty = first.confirmed(by: DetailRead(lines: [], missedLines: 0, droppedNumberLines: 0))
    expect(empty.lines == ["식후에 드세요"] && empty.droppedNumberLines == 2, "no second read keeps no number lines")
}
