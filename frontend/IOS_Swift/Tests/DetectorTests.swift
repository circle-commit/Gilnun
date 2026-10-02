import CoreImage
import CoreVideo
import Foundation

private struct GoldenCase: Decodable {
    let image: String
    let crop: [Int]
    let detections: [GoldenDetection]
}

private struct GoldenDetection: Decodable {
    let label: String
    let confidence: Double
    let bbox: [Double]
}

/// Runs the Core ML detector on the golden portrait crops and checks that it finds
/// the same objects as the PyTorch checkpoint (see make_detector_golden.py).
///
/// Each crop is checked twice: decoded straight from the JPEG, and through the app's
/// camera path (a landscape YUV sensor buffer rotated by `ObjectDetector.uprightFrame`).
func runDetectorTests(modelURL: URL, goldenURL: URL, repoRoot: URL) {
    let detector: ObjectDetector
    let cases: [GoldenCase]
    do {
        detector = try ObjectDetector(compiledModelURL: modelURL)
        cases = try JSONDecoder().decode([GoldenCase].self, from: Data(contentsOf: goldenURL))
    } catch {
        expect(false, "could not set up detector test: \(error)")
        return
    }

    let context = CIContext()
    let paths: [(name: String, prepare: (CIImage) -> CIImage?)] = [
        ("jpeg", { $0 }),
        ("camera", { cameraFrame(from: $0, context: context) }),
    ]

    for path in paths {
        var stats = MatchStats()
        for testCase in cases {
            guard let image = CIImage(contentsOf: repoRoot.appendingPathComponent(testCase.image)) else {
                expect(false, "could not load \(testCase.image)")
                continue
            }

            // Golden crops use top-left pixel coordinates; Core Image's origin is bottom-left.
            let x = Double(testCase.crop[0])
            let y = Double(testCase.crop[1])
            let width = Double(testCase.crop[2])
            let height = Double(testCase.crop[3])
            let crop = image.cropped(to: CGRect(x: x, y: image.extent.height - y - height, width: width, height: height))

            guard let frame = path.prepare(crop) else {
                expect(false, "\(path.name) \(testCase.image): could not build frame")
                continue
            }
            expect(
                frame.extent.width == width && frame.extent.height == height,
                "\(path.name) \(testCase.image): frame is \(frame.extent.size), expected upright \(width)x\(height)"
            )

            do {
                let detections = try detector.detect(in: frame)
                stats.compare(detections, with: testCase.detections, threshold: detector.confidenceThreshold, name: "\(path.name) \(testCase.image)")
            } catch {
                expect(false, "\(path.name) \(testCase.image): detection failed: \(error)")
            }
        }

        expect(stats.largestConfidenceGap <= 0.05, "\(path.name): confidence differs by \(stats.largestConfidenceGap)")
        print(
            "detector (\(path.name)): matched \(stats.matched)/\(stats.expected) PyTorch detections, " +
            "max confidence gap \(String(format: "%.4f", stats.largestConfidenceGap)), " +
            "min IoU \(String(format: "%.3f", stats.smallestIoU))"
        )
    }
}

private struct MatchStats {
    var matched = 0
    var expected = 0
    var largestConfidenceGap = 0.0
    var smallestIoU = 1.0

    mutating func compare(_ detections: [RawDetection], with golden: [GoldenDetection], threshold: Double, name: String) {
        // Objects this close to the threshold may legitimately flip between runtimes.
        let borderline = threshold + 0.05
        var used = Set<Int>()

        for reference in golden {
            expected += 1
            let referenceBox = BoundingBox(x1: reference.bbox[0], y1: reference.bbox[1], x2: reference.bbox[2], y2: reference.bbox[3])
            let iou = { (index: Int) in detections[index].bbox.intersectionOverUnion(with: referenceBox) }
            let best = detections.indices
                .filter { !used.contains($0) && detections[$0].label == reference.label }
                .max { iou($0) < iou($1) }

            if let best, iou(best) >= 0.8 {
                used.insert(best)
                matched += 1
                largestConfidenceGap = max(largestConfidenceGap, abs(detections[best].confidence - reference.confidence))
                smallestIoU = min(smallestIoU, iou(best))
            } else {
                expect(reference.confidence < borderline, "\(name): missed \(reference.label) (\(reference.confidence))")
            }
        }

        for index in detections.indices where !used.contains(index) {
            expect(detections[index].confidence < borderline, "\(name): unexpected \(detections[index].label) (\(detections[index].confidence))")
        }
    }
}

/// Simulates what the camera delivers: the upright crop rotated into the sensor's
/// landscape orientation, stored as a full-range 4:2:0 YUV buffer, then turned
/// upright again by the same function the app uses.
private func cameraFrame(from upright: CIImage, context: CIContext) -> CIImage? {
    let origin = CGAffineTransform(translationX: -upright.extent.minX, y: -upright.extent.minY)
    let rotated = upright.transformed(by: origin).oriented(.left)
    let sensorImage = rotated.transformed(by: CGAffineTransform(translationX: -rotated.extent.minX, y: -rotated.extent.minY))

    var buffer: CVPixelBuffer?
    let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any]] as CFDictionary
    guard CVPixelBufferCreate(
        kCFAllocatorDefault,
        Int(sensorImage.extent.width),
        Int(sensorImage.extent.height),
        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        attributes,
        &buffer
    ) == kCVReturnSuccess, let buffer else {
        return nil
    }

    CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
    context.render(sensorImage, to: buffer)

    return ObjectDetector.uprightFrame(from: buffer)
}
