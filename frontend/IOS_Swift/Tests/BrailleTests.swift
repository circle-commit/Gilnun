import CoreImage
import Foundation

private struct BrailleGoldenCase: Decodable {
    let image: String
    let crop: [Int]
    let rows: [String]
}

/// Runs the Core ML segmenter on portrait crops and checks its mask against the PyTorch
/// checkpoint's (see make_braille_golden.py), straight from the JPEG and through the
/// camera path. The images are AI Hub data kept out of the repository; missing ones are skipped.
func runBrailleSegmenterTests(modelURL: URL, goldenURL: URL, repoRoot: URL) {
    let segmenter: BrailleBlockSegmenter
    let cases: [BrailleGoldenCase]
    do {
        segmenter = try BrailleBlockSegmenter(compiledModelURL: modelURL)
        cases = try JSONDecoder().decode([BrailleGoldenCase].self, from: Data(contentsOf: goldenURL))
    } catch {
        expect(false, "could not set up braille segmenter test: \(error)")
        return
    }

    let context = CIContext()
    var checked = 0
    var smallestIoU = 1.0
    for testCase in cases {
        guard let image = CIImage(contentsOf: repoRoot.appendingPathComponent(testCase.image)) else { continue }
        let x = Double(testCase.crop[0]), y = Double(testCase.crop[1])
        let width = Double(testCase.crop[2]), height = Double(testCase.crop[3])
        let crop = image.cropped(to: CGRect(x: x, y: image.extent.height - y - height, width: width, height: height))
        let golden = testCase.rows.joined().map { $0 == "1" }
        let goldenMask = BrailleMask(cells: golden)

        for (name, frame) in [("jpeg", crop), ("camera", cameraFrame(from: crop, context: context))] {
            guard let frame, let mask = try? segmenter.segment(in: frame) else {
                expect(false, "\(name) \(testCase.image): segmentation failed")
                continue
            }
            checked += 1
            let both = zip(mask.cells, golden).filter { $0 && $1 }.count
            let either = zip(mask.cells, golden).filter { $0 || $1 }.count
            let goldenCount = golden.filter { $0 }.count
            if goldenCount >= 20 {
                let iou = Double(both) / Double(max(1, either))
                smallestIoU = min(smallestIoU, iou)
                // A small mask's IoU is mostly its edge cells, which shift with the camera's YUV colors.
                let required = goldenCount >= 100 ? 0.85 : 0.6
                expect(iou >= required, "\(name) \(testCase.image): mask IoU \(iou) on \(goldenCount) cells")
            } else {
                expect(either - both <= 10, "\(name) \(testCase.image): \(either - both) cells differ on a nearly empty mask")
            }
            expect(
                BrailleBlockTracker.position(of: mask) == BrailleBlockTracker.position(of: goldenMask),
                "\(name) \(testCase.image): side differs from PyTorch"
            )
        }
    }
    print(checked == 0
        ? "braille segmenter: skipped (datasets/braille_seg not found)"
        : "braille segmenter: \(checked) frames match PyTorch, min IoU \(String(format: "%.3f", smallestIoU))")
}

func runBrailleTrackerTests() {
    /// Blocks filling the given columns in the given rows.
    func mask(columns: Range<Int>, rows: Range<Int> = 40..<80) -> BrailleMask {
        var mask = BrailleMask.empty
        for row in rows {
            for column in columns {
                mask.cells[row * BrailleMask.columns + column] = true
            }
        }
        return mask
    }
    let left = mask(columns: 2..<8)
    let center = mask(columns: 19..<26)

    // Announced once, after three frames in a row.
    var tracker = BrailleBlockTracker()
    expect(tracker.update(left, now: 0, canSpeak: true) == nil, "braille: spoke after one frame")
    expect(tracker.update(left, now: 0.2, canSpeak: true) == nil, "braille: spoke after two frames")
    expect(tracker.update(left, now: 0.4, canSpeak: true) == "왼쪽에 점자블록이 있어요.", "braille: left not announced")
    expect(tracker.update(left, now: 0.6, canSpeak: true) == nil, "braille: repeated itself")

    // A new side waits for the side-change interval.
    _ = tracker.update(center, now: 1.0, canSpeak: true)
    _ = tracker.update(center, now: 1.2, canSpeak: true)
    expect(tracker.update(center, now: 1.4, canSpeak: true) == nil, "braille: side change too soon")
    expect(tracker.update(center, now: 4.4, canSpeak: true) == "앞에 점자블록이 있어요.", "braille: center not announced")

    // Gone for two seconds, then back: announced again.
    for frame in 0..<BrailleBlockTracker.framesToLose {
        expect(tracker.update(.empty, now: 5 + Double(frame) * 0.2, canSpeak: true) == nil, "braille: spoke without blocks")
    }
    for frame in 0..<3 {
        let message = tracker.update(center, now: 8 + Double(frame) * 0.2, canSpeak: true)
        expect((frame == 2) == (message != nil), "braille: return after loss, frame \(frame) -> \(String(describing: message))")
    }

    // Urgent speech defers the announcement instead of dropping it.
    tracker = BrailleBlockTracker()
    for frame in 0..<4 {
        expect(tracker.update(left, now: Double(frame) * 0.2, canSpeak: false) == nil, "braille: spoke over urgent guidance")
    }
    expect(tracker.update(left, now: 1, canSpeak: true) == "왼쪽에 점자블록이 있어요.", "braille: deferred announcement lost")

    // Flicker (a false positive every other frame) never confirms.
    tracker = BrailleBlockTracker()
    for frame in 0..<20 {
        expect(tracker.update(frame % 2 == 0 ? center : .empty, now: Double(frame) * 0.2, canSpeak: true) == nil, "braille: flicker announced")
    }

    // Blocks far ahead (upper frame) or a speck are not in the walking area.
    expect(BrailleBlockTracker.position(of: mask(columns: 0..<45, rows: 0..<30)) == nil, "braille: far blocks counted")
    expect(BrailleBlockTracker.position(of: mask(columns: 40..<41, rows: 70..<80)) == nil, "braille: speck counted")
    expect(BrailleBlockTracker.position(of: mask(columns: 38..<44)) == .right, "braille: right side")
}
