import AVFoundation
import CoreGraphics
import Foundation

/// One LiDAR depth frame: distances in meters, in the camera sensor's landscape
/// orientation (the orientation of the video buffers before `ObjectDetector.uprightFrame`
/// turns them upright). It covers the same field of view as the video frame.
nonisolated struct DepthMap {
    let width: Int
    let height: Int
    /// Row-major distances in meters; zero, negative or non-finite values mean no reading.
    let meters: [Float]

    /// The iPhone LiDAR measures reliably up to about 5 m.
    static let maximumRange = 5.0

    /// Distance to the object inside `box`, or nil when the LiDAR has no reliable reading
    /// there (too few valid points, or farther than `maximumRange`).
    ///
    /// - Parameters:
    ///   - box: Detector box in upright frame pixels (origin top-left).
    ///   - uprightFrameSize: Size of the upright frame the box is measured in.
    func distance(to box: BoundingBox, uprightFrameSize: CGSize) -> Double? {
        let frameWidth = Double(uprightFrameSize.width)
        let frameHeight = Double(uprightFrameSize.height)
        guard width > 0, height > 0, meters.count == width * height, frameWidth > 0, frameHeight > 0 else {
            return nil
        }

        // `uprightFrame` turns the sensor image 90° clockwise (`.right`), so the upright
        // point (u, v) comes from the sensor point (v, 1 - u) in normalized coordinates.
        var sensorX1 = box.y1 / frameHeight
        var sensorX2 = box.y2 / frameHeight
        var sensorY1 = 1 - box.x2 / frameWidth
        var sensorY2 = 1 - box.x1 / frameWidth

        // Use the inner half of the box: its edges usually show the background behind the object.
        let insetX = (sensorX2 - sensorX1) / 4
        let insetY = (sensorY2 - sensorY1) / 4
        sensorX1 += insetX
        sensorX2 -= insetX
        sensorY1 += insetY
        sensorY2 -= insetY

        let columns = pixelRange(from: sensorX1, to: sensorX2, size: width)
        let rows = pixelRange(from: sensorY1, to: sensorY2, size: height)
        guard !columns.isEmpty, !rows.isEmpty else { return nil }

        var readings: [Float] = []
        readings.reserveCapacity(columns.count * rows.count)
        for row in rows {
            let rowStart = row * width
            for column in columns {
                let value = meters[rowStart + column]
                if value.isFinite && value > 0 {
                    readings.append(value)
                }
            }
        }

        // Too many holes means the reading is not about this object.
        guard readings.count >= 4, readings.count * 4 >= columns.count * rows.count else { return nil }

        // A low percentile leans toward the object, which is usually in front of whatever
        // shows through it (the gaps in a bicycle, the space under a bench).
        readings.sort()
        // Centimeters are as fine as the LiDAR resolves.
        let distance = (Double(readings[Int(Double(readings.count - 1) * 0.3)]) * 100).rounded() / 100
        return distance <= Self.maximumRange ? distance : nil
    }

    private func pixelRange(from start: Double, to end: Double, size: Int) -> Range<Int> {
        let lower = max(0, min(size, Int((start * Double(size)).rounded(.down))))
        let upper = max(lower, min(size, Int((end * Double(size)).rounded(.up))))
        return lower..<upper
    }
}

extension DepthMap {
    /// Copies a LiDAR depth frame out of the capture buffer. Returns nil for relative
    /// (non-metric) depth, which cannot be spoken as meters.
    nonisolated init?(depthData: AVDepthData) {
        guard depthData.depthDataAccuracy == .absolute else { return nil }

        let depth = depthData.depthDataType == kCVPixelFormatType_DepthFloat32
            ? depthData
            : depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let buffer = depth.depthDataMap
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }

        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        var meters = [Float](repeating: 0, count: width * height)
        for row in 0..<height {
            let source = base.advanced(by: row * bytesPerRow).assumingMemoryBound(to: Float32.self)
            for column in 0..<width {
                meters[row * width + column] = source[column]
            }
        }
        self.init(width: width, height: height, meters: meters)
    }
}
