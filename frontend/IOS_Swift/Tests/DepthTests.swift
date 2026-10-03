import CoreGraphics
import CoreImage
import CoreVideo
import Foundation

// MARK: - LiDAR distance

func runDepthTests() {
    runDepthOrientationTest()
    runDepthSamplingTests()
    runDistanceGuidanceTests()
}

/// The depth map is in the sensor's landscape orientation, like the video buffer. Draws a
/// block into a sensor-oriented buffer, turns it upright the way the app does, finds the
/// block in the upright image and checks that its box reads the block's depth.
private func runDepthOrientationTest() {
    // The pixel reader must report Core Image's top (largest y) as row 0, like detector boxes.
    let white = CIImage(color: CIColor(red: 1, green: 1, blue: 1)).cropped(to: CGRect(x: 0, y: 0, width: 10, height: 20))
    let topBand = CIImage(color: CIColor(red: 0, green: 0, blue: 0)).cropped(to: CGRect(x: 0, y: 16, width: 10, height: 4))
    let band = darkBox(in: topBand.composited(over: white))
    expect(band == BoundingBox(x1: 0, y1: 0, x2: 10, y2: 4), "depth orientation: reader puts the image top at row 0, got \(String(describing: band))")

    let sensorWidth = 64
    let sensorHeight = 36
    let blockColumns = 8..<20
    let blockRows = 6..<14

    guard let buffer = makeSensorBuffer(width: sensorWidth, height: sensorHeight, columns: blockColumns, rows: blockRows) else {
        expect(false, "depth orientation: could not create a pixel buffer")
        return
    }
    let upright = ObjectDetector.uprightFrame(from: buffer)
    let uprightSize = upright.extent.size
    expect(uprightSize == CGSize(width: sensorHeight, height: sensorWidth), "depth orientation: upright frame is portrait")

    guard let box = darkBox(in: upright) else {
        expect(false, "depth orientation: block not found in the upright frame")
        return
    }

    var meters = [Float](repeating: 4.5, count: sensorWidth * sensorHeight)
    for row in blockRows {
        for column in blockColumns {
            meters[row * sensorWidth + column] = 2.0
        }
    }
    let depth = DepthMap(width: sensorWidth, height: sensorHeight, meters: meters)
    let distance = depth.distance(to: box, uprightFrameSize: uprightSize)
    expect(distance == 2.0, "depth orientation: box over the block reads 2.0 m, got \(String(describing: distance))")

    // The same box mirrored onto the other side of the frame must miss the block.
    let mirrored = BoundingBox(
        x1: uprightSize.width - box.x2,
        y1: box.y1,
        x2: uprightSize.width - box.x1,
        y2: box.y2
    )
    expect(depth.distance(to: mirrored, uprightFrameSize: uprightSize) == 4.5, "depth orientation: mirrored box reads the background")
}

private func runDepthSamplingTests() {
    let width = 320
    let height = 180
    let frame = CGSize(width: 1080, height: 1920)
    // Upright box over the middle of the frame; in sensor space that is columns 80..<240 and rows 45..<135.
    let box = BoundingBox(x1: 270, y1: 480, x2: 810, y2: 1440)

    func map(_ value: (Int, Int) -> Float) -> DepthMap {
        var meters = [Float](repeating: 0, count: width * height)
        for row in 0..<height {
            for column in 0..<width {
                meters[row * width + column] = value(column, row)
            }
        }
        return DepthMap(width: width, height: height, meters: meters)
    }

    let solid = map { _, _ in 1.8 }
    expect(solid.distance(to: box, uprightFrameSize: frame) == 1.8, "depth: solid object reads its distance")

    // A bicycle-like object: the background shows through every other column.
    let gaps = map { column, _ in column % 2 == 0 ? 1.8 : 4.8 }
    expect(gaps.distance(to: box, uprightFrameSize: frame) == 1.8, "depth: gaps in the object do not pull the distance back")

    let empty = map { _, _ in 0 }
    expect(empty.distance(to: box, uprightFrameSize: frame) == nil, "depth: no readings gives nil")

    let tooFar = map { _, _ in 7.0 }
    expect(tooFar.distance(to: box, uprightFrameSize: frame) == nil, "depth: beyond the LiDAR range gives nil")

    let mostlyHoles = map { column, row in column % 5 == 0 && row % 2 == 0 ? 1.8 : .nan }
    expect(mostlyHoles.distance(to: box, uprightFrameSize: frame) == nil, "depth: mostly holes gives nil")

    let wrongSize = DepthMap(width: width, height: height, meters: [1.0])
    expect(wrongSize.distance(to: box, uprightFrameSize: frame) == nil, "depth: malformed map gives nil")

    // The analyzer attaches the reading to its detection.
    let analyzer = SceneAnalyzer()
    let result = analyzer.analyze(
        [RawDetection(label: "person", confidence: 0.9, bbox: box)],
        frameSize: frame,
        now: 10,
        depth: solid
    )
    expect(result.detections.first?.distanceMeters == 1.8, "depth: analyzer sets distanceMeters")
    let withoutDepth = SceneAnalyzer().analyze([RawDetection(label: "person", confidence: 0.9, bbox: box)], frameSize: frame, now: 10)
    expect(withoutDepth.detections.first?.distanceMeters == nil, "depth: no depth map leaves distanceMeters nil")
}

private func runDistanceGuidanceTests() {
    let levels: [(Double, DistanceLevel)] = [
        (0.8, .veryClose), (1.19, .veryClose), (1.2, .close), (2.49, .close), (2.5, .near), (4.99, .near), (5.0, .far),
    ]
    for (meters, level) in levels {
        expect(GuidanceRules.distanceLevel(meters: meters) == level, "distance level for \(meters) m is \(level)")
    }

    let phrases: [(Double, String)] = [
        (0.6, "1미터 안이에요."), (1.26, "약 1.5미터 거리예요."), (2.04, "약 2미터 거리예요."),
        (2.8, "약 3미터 거리예요."), (3.4, "약 3미터 거리예요."), (4.6, "약 5미터 거리예요."),
    ]
    for (meters, phrase) in phrases {
        expect(GuidanceRules.distancePhrase(meters: meters) == phrase, "phrase for \(meters) m is \(phrase)")
    }

    // A measured distance replaces the box-size estimate: a big car box 4 m away is only near.
    var car = depthDetection("car", position: .center, areaRatio: 0.2, verticalRatio: 0.95)
    expect(GuidanceRules.distanceLevel(of: car) == .veryClose, "car box estimate without LiDAR is very close")
    car.distanceMeters = 4.0
    expect(GuidanceRules.distanceLevel(of: car) == .near, "car 4 m away by LiDAR is near")

    var bench = depthDetection("bench", position: .left, areaRatio: 0.05, verticalRatio: 0.7)
    let plain = GuidanceRules.detectionMessage(for: bench, eventType: .newObject) { _ in 0 }
    expect(!plain.contains("미터"), "message without LiDAR has no distance: \(plain)")
    bench.distanceMeters = 2.04
    let measured = GuidanceRules.detectionMessage(for: bench, eventType: .newObject) { _ in 0 }
    expect(measured.hasSuffix(" 약 2미터 거리예요."), "message with LiDAR ends with the distance: \(measured)")

    // Announced again once the object is a meter closer than when it was announced.
    let tracker = GuidanceEventTracker(cooldowns: .init(global: 0, object: 0, situation: 0, signature: 0)) { _ in 0 }
    var now = 50.0
    func frame(_ meters: Double) -> [GuidanceEvent] {
        now += 0.2
        var detection = depthDetection("bench", position: .left, areaRatio: 0.05, verticalRatio: 0.7)
        detection.distanceMeters = meters
        return tracker.chooseEvents(GuidanceRules.enrichAndPrioritize([detection]), now: now)
    }
    expect(frame(4.0).isEmpty, "closer: first sighting waits for confirmation")
    expect(frame(4.0).map(\.type) == [.newObject], "closer: second sighting is announced")
    expect(frame(3.5).isEmpty, "closer: half a meter closer is not announced")
    let closer = frame(2.8)
    expect(closer.map(\.type) == [.closer], "closer: a meter closer is announced, got \(closer.map(\.type))")
    expect(closer.first?.message.hasSuffix("약 3미터 거리예요.") == true, "closer: message has the new distance")
    expect(frame(2.6).isEmpty, "closer: measured from the last announcement")
}

private func depthDetection(
    _ label: String,
    position: HorizontalPosition,
    areaRatio: Double,
    verticalRatio: Double
) -> SceneDetection {
    SceneDetection(
        label: label,
        koreanLabel: SceneAnalyzer.koreanLabels[label] ?? label,
        confidence: 0.8,
        bbox: BoundingBox(x1: 100, y1: 900, x2: 300, y2: 1300),
        frameWidth: 1080,
        position: position,
        areaRatio: areaRatio,
        verticalRatio: verticalRatio
    )
}

/// White BGRA buffer in sensor orientation with a black block.
private func makeSensorBuffer(width: Int, height: Int, columns: Range<Int>, rows: Range<Int>) -> CVPixelBuffer? {
    var buffer: CVPixelBuffer?
    let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any]] as CFDictionary
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attributes, &buffer) == kCVReturnSuccess,
          let buffer else { return nil }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    for row in 0..<height {
        let pixels = base.advanced(by: row * bytesPerRow).assumingMemoryBound(to: UInt8.self)
        for column in 0..<width {
            let value: UInt8 = rows.contains(row) && columns.contains(column) ? 0 : 255
            pixels[column * 4] = value
            pixels[column * 4 + 1] = value
            pixels[column * 4 + 2] = value
            pixels[column * 4 + 3] = 255
        }
    }
    return buffer
}

/// Box (origin top-left, like detector boxes) around the dark pixels of an image.
private func darkBox(in image: CIImage) -> BoundingBox? {
    let context = CIContext(options: [.cacheIntermediates: false])
    let extent = image.extent
    guard let cgImage = context.createCGImage(image, from: extent) else { return nil }

    let width = cgImage.width
    let height = cgImage.height
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    let drawn = pixels.withUnsafeMutableBytes { raw -> Bool in
        guard let bitmap = CGContext(
            data: raw.baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return false }
        bitmap.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return true
    }
    guard drawn else { return nil }

    // Bitmap memory starts with the top row of the image.
    var minX = width, minY = height, maxX = -1, maxY = -1
    for y in 0..<height {
        for x in 0..<width where pixels[(y * width + x) * 4] < 128 {
            minX = min(minX, x)
            maxX = max(maxX, x)
            minY = min(minY, y)
            maxY = max(maxY, y)
        }
    }
    guard maxX >= 0 else { return nil }
    return BoundingBox(x1: Double(minX), y1: Double(minY), x2: Double(maxX + 1), y2: Double(maxY + 1))
}
