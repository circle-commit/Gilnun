package com.kojingapla.glass

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.util.TypedValue
import android.view.GestureDetector
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.accessibility.AccessibilityManager
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

/**
 * Native Android port of the iOS Gilnun app. Everything runs on the phone, like on iOS:
 * the sidewalk YOLO model (ONNX Runtime) finds obstacles for live guidance, and ML Kit's
 * on-device Korean recognizer reads text. Camera frames never leave the device.
 *
 * Layout mirrors the earlier SwiftUI design: a status bar, a mode pill, a bounding-box
 * overlay over the live preview, a guidance card (live) / OCR panel (text) and a bottom
 * mode bar. Double tap anywhere switches modes; in text mode a single tap repeats the
 * last text.
 */
class MainActivity : ComponentActivity() {

    // ── Business logic ────────────────────────────────────────
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private lateinit var speech: SpeechManager
    private lateinit var haptics: HapticFeedbackManager
    private var camera: Camera? = null

    @Volatile private var currentMode = ProcessingMode.LIVE

    // Camera thread only.
    private val sceneAnalyzer = SceneAnalyzer()
    private val stabilityTracker = TextStabilityTracker()
    private val duplicateSuppressor = DuplicateTextSuppressor()
    private var objectDetector: ObjectDetector? = null
    private var objectDetectorFailed = false
    private val textRecognizer by lazy { TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build()) }
    private var lastLiveAnalysisMs = 0L
    private var lastLiveGuideMs = 0L
    private var displayedRiskScore = 0
    private var displayedRiskMs = 0L
    private var lastTextAnalysisMs = 0L
    private var previousLuma: LumaSample? = null
    private var lastFullOcrMs = Long.MIN_VALUE / 2
    /** Consecutive dark text-mode frames. */
    private var darkFrameCount = 0
    /** The first of two close reads; digits are spoken only when the next frame reads the same. */
    private var pendingDetail: Pair<DetailRead, Long>? = null
    @Volatile private var isTorchOn = false

    // Main thread only.
    private var latestGuide = "실시간 안내 모드가 준비되었습니다."
    private var latestDetectedText: String? = null
    private var liveOcrStatus = LiveOcrStatus.SEARCHING
    private var latestLiveRiskScore = 0
    private var isLiveAnalysisRunning = false
    private var isCameraDenied = false
    private var liveBoxes: List<LiveGuidanceBox> = emptyList()
    private var liveImageW = 0f
    private var liveImageH = 0f

    // ── UI refs ───────────────────────────────────────────────
    private lateinit var previewView: PreviewView
    private lateinit var boxOverlay: BoundingBoxOverlayView
    private lateinit var statusDot: View
    private lateinit var modeLabel: TextView
    private lateinit var modeSpinner: ProgressBar
    private lateinit var guidanceCard: LinearLayout
    private lateinit var guidanceAccent: View
    private lateinit var guidanceTag: TextView
    private lateinit var guidanceMsg: TextView
    private lateinit var ocrCard: LinearLayout
    private lateinit var ocrStatusLbl: TextView
    private lateinit var ocrSpinner: ProgressBar
    private lateinit var ocrDivider: View
    private lateinit var ocrResultLbl: TextView
    private lateinit var btnLive: LinearLayout
    private lateinit var btnLiveIcon: IconView
    private lateinit var btnLiveLabel: TextView
    private lateinit var btnOcr: LinearLayout
    private lateinit var btnOcrIcon: IconView
    private lateinit var btnOcrLabel: TextView

    private val density get() = resources.displayMetrics.density

    // ── Lifecycle ─────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        speech = SpeechManager(this)
        haptics = HapticFeedbackManager(this)
        buildUi()
        // The model takes a moment to load; start before the camera is up.
        cameraExecutor.execute { loadObjectDetectorIfNeeded() }
        if (hasCameraPermission()) startCameraWhenReady()
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST)
    }

    override fun onRequestPermissionsResult(req: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(req, perms, results)
        if (req != CAMERA_REQUEST) return
        if (results.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            startCameraWhenReady()
        } else {
            isCameraDenied = true
            liveOcrStatus = LiveOcrStatus.UNAVAILABLE
            latestGuide = "이 앱을 사용하려면 카메라 권한이 필요합니다."
            renderState()
        }
    }

    override fun onStop() {
        super.onStop()
        haptics.stopRepeatingPulses()
        speech.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.execute {
            objectDetector?.close()
            objectDetector = null
            textRecognizer.close()
        }
        cameraExecutor.shutdown()
        haptics.stopRepeatingPulses()
        speech.shutdown()
    }

    // ── UI build ──────────────────────────────────────────────
    private fun buildUi() {
        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        boxOverlay = BoundingBoxOverlayView(this)

        val root = FrameLayout(this)
        root.addView(previewView, matchParent())
        root.addView(makeVignette(), matchParent())
        root.addView(boxOverlay, matchParent())
        root.addView(makeGestureLayer(), matchParent())

        val overlay = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(overlay, matchParent())

        // Edge-to-edge: pad the overlay by the system-bar insets so the camera bleeds
        // full-screen behind a status bar / nav bar that never overlaps the controls.
        ViewCompat.setOnApplyWindowInsetsListener(overlay) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(0, bars.top, 0, bars.bottom)
            insets
        }

        overlay.addView(makeStatusBar(), rowLp(top = 14, h = 36))
        overlay.addView(makeModePill(), rowLp(top = 16, h = 46))
        overlay.addView(View(this), LinearLayout.LayoutParams(MATCH, 0, 1f))   // Spacer
        guidanceCard = makeGuidanceCard()
        overlay.addView(guidanceCard, cardLp())
        ocrCard = makeOcrCard()
        overlay.addView(ocrCard, cardLp())
        overlay.addView(makeModeBar(), modeBarLp())

        setContentView(root)
        renderState()
    }

    /**
     * Anyone who can't find the mode buttons can switch from anywhere on screen with a
     * double tap; in text mode a single tap repeats the last text. Touches on the cards
     * and buttons never reach this layer.
     */
    @SuppressLint("ClickableViewAccessibility")
    private fun makeGestureLayer(): View {
        val detector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent): Boolean = true

            override fun onDoubleTap(e: MotionEvent): Boolean {
                toggleMode()
                return true
            }

            override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                if (currentMode == ProcessingMode.TEXT) repeatLastText()
                return true
            }
        })
        return View(this).apply {
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
            setOnTouchListener { _, event -> detector.onTouchEvent(event) }
        }
    }

    // Vignette: top dark → clear → bottom dark, matching iOS VignetteLayer (bg #070810).
    private fun makeVignette(): View = View(this).apply {
        importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        background = GradientDrawable(
            GradientDrawable.Orientation.TOP_BOTTOM,
            intArrayOf(0xD1070810.toInt(), 0x2E070810, 0xEB070810.toInt())
        )
    }

    // ── Status bar ────────────────────────────────────────────
    private fun makeStatusBar(): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val icon = IconView(this, IconView.Type.CAMERA_VIEWFINDER).apply { setColor(C_PRIMARY) }
        row.addView(icon, LinearLayout.LayoutParams(dp(22), dp(22)))
        val title = TextView(this).apply {
            text = getString(R.string.app_name)
            textSize = 18f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            setTextColor(Color.WHITE)
            setPadding(dp(10), 0, 0, 0)
        }
        row.addView(title)
        return row
    }

    // ── Mode pill ─────────────────────────────────────────────
    private fun makeModePill(): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = capsule(C_PILL, C_GLASS_STROKE, 23)
        }
        statusDot = View(this).apply { background = circle(C_LIVE) }
        row.addView(statusDot, LinearLayout.LayoutParams(dp(10), dp(10)).apply { marginStart = dp(14) })

        modeLabel = TextView(this).apply {
            text = "보행 안내"
            textSize = 17f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            setTextColor(Color.WHITE)
            setPadding(dp(9), 0, 0, 0)
        }
        row.addView(modeLabel)

        modeSpinner = spinner(0.75f)
        row.addView(modeSpinner, LinearLayout.LayoutParams(dp(24), dp(24)).apply { marginStart = dp(10) })

        row.addView(View(this), LinearLayout.LayoutParams(0, 1, 1f))   // trailing Spacer
        return row
    }

    // ── Guidance card (live) ──────────────────────────────────
    private fun makeGuidanceCard(): LinearLayout {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(22), dp(22), dp(22), dp(22))
            background = glass(C_GLASS, withAlpha(C_PRIMARY, 0.28f), 24)
            elevation = dp(12).toFloat()
        }

        val tagRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        guidanceAccent = View(this).apply { background = rounded(C_PRIMARY, 3) }
        tagRow.addView(guidanceAccent, LinearLayout.LayoutParams(dp(5), dp(20)))
        guidanceTag = TextView(this).apply {
            text = "안내"
            textSize = 14f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            letterSpacing = 0.08f
            setTextColor(C_PRIMARY)
            setPadding(dp(8), 0, 0, 0)
        }
        tagRow.addView(guidanceTag)
        card.addView(tagRow)

        guidanceMsg = TextView(this).apply {
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            setTextColor(Color.WHITE)
            setLineSpacing(dp(5).toFloat(), 1f)
            maxLines = 3
            // 28sp, shrinking to ~0.75× (≈21sp) to fit — mirrors iOS minimumScaleFactor(0.75).
            setAutoSizeTextTypeUniformWithConfiguration(21, 28, 1, TypedValue.COMPLEX_UNIT_SP)
        }
        card.addView(guidanceMsg, LinearLayout.LayoutParams(MATCH, WRAP).apply { topMargin = dp(12) })
        return card
    }

    // ── OCR panel (text) ──────────────────────────────────────
    private fun makeOcrCard(): LinearLayout {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(22), dp(22), dp(22), dp(22))
            background = glass(C_GLASS, withAlpha(C_PRIMARY, 0.25f), 24)
            elevation = dp(18).toFloat()
            // A tap on the card, or a double tap with TalkBack, reads the text again.
            isClickable = true
            setOnClickListener { repeatLastText() }
        }

        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val iconBox = FrameLayout(this).apply { background = rounded(withAlpha(C_PRIMARY, 0.14f), 10) }
        val icon = IconView(this, IconView.Type.TEXT_VIEWFINDER).apply { setColor(C_PRIMARY) }
        iconBox.addView(icon, FrameLayout.LayoutParams(dp(20), dp(20), Gravity.CENTER))
        header.addView(iconBox, LinearLayout.LayoutParams(dp(36), dp(36)))

        ocrStatusLbl = TextView(this).apply {
            text = liveOcrStatus.message
            textSize = 17f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            setTextColor(C_DIM_TEXT)
            setPadding(dp(12), 0, 0, 0)
        }
        header.addView(ocrStatusLbl, LinearLayout.LayoutParams(0, WRAP, 1f))

        ocrSpinner = spinner(0.85f)
        header.addView(ocrSpinner, LinearLayout.LayoutParams(dp(26), dp(26)))
        card.addView(header)

        ocrDivider = View(this).apply { setBackgroundColor(0x14FFFFFF) }
        card.addView(ocrDivider, LinearLayout.LayoutParams(MATCH, dp(1)).apply {
            topMargin = dp(16); bottomMargin = dp(16)
        })

        ocrResultLbl = TextView(this).apply {
            textSize = 32f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            setTextColor(Color.WHITE)
            setLineSpacing(dp(6).toFloat(), 1f)
        }
        card.addView(ocrResultLbl, LinearLayout.LayoutParams(MATCH, WRAP))
        return card
    }

    // ── Mode bar (bottom) ─────────────────────────────────────
    private fun makeModeBar(): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(7), dp(7), dp(7), dp(7))
            background = glass(C_GLASS, C_GLASS_STROKE, 26)
            elevation = dp(12).toFloat()
        }

        val live = makeModeButton(IconView.Type.EYE, "실시간") { switchMode(ProcessingMode.LIVE) }
        btnLive = live.first; btnLiveIcon = live.second; btnLiveLabel = live.third
        val ocr = makeModeButton(IconView.Type.TEXT_VIEWFINDER, "문자 읽기") { switchMode(ProcessingMode.TEXT) }
        btnOcr = ocr.first; btnOcrIcon = ocr.second; btnOcrLabel = ocr.third

        row.addView(btnLive, LinearLayout.LayoutParams(0, dp(58), 1f))
        row.addView(btnOcr, LinearLayout.LayoutParams(0, dp(58), 1f).apply { marginStart = dp(8) })
        return row
    }

    private fun makeModeButton(
        iconType: IconView.Type,
        label: String,
        onClick: () -> Unit
    ): Triple<LinearLayout, IconView, TextView> {
        val icon = IconView(this, iconType)
        val text = TextView(this).apply {
            this.text = label
            textSize = 18f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            gravity = Gravity.CENTER
            setPadding(dp(8), 0, 0, 0)
        }
        val btn = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            background = rounded(C_PILL, 16)
            isClickable = true
            isFocusable = true
            contentDescription = label
            setOnClickListener { onClick() }
            addView(icon, LinearLayout.LayoutParams(dp(22), dp(22)))
            addView(text)
        }
        return Triple(btn, icon, text)
    }

    // ── Render ────────────────────────────────────────────────
    private fun renderState() {
        val isOcr = currentMode == ProcessingMode.TEXT

        statusDot.background = circle(
            when {
                isCameraDenied -> C_DANGER
                isOcr -> C_PRIMARY
                isLiveAnalysisRunning -> C_LIVE
                else -> C_DIM_TEXT
            }
        )
        modeLabel.text = when {
            isCameraDenied -> "카메라 권한이 필요해요"
            isOcr -> "문자 읽기"
            isLiveAnalysisRunning -> "주변을 살피는 중"
            else -> "카메라 준비 중"
        }
        modeSpinner.visibility = if (!isOcr && !isLiveAnalysisRunning && !isCameraDenied) View.VISIBLE else View.GONE

        guidanceCard.visibility = if (isOcr || latestGuide.isEmpty()) View.GONE else View.VISIBLE
        ocrCard.visibility = if (isOcr) View.VISIBLE else View.GONE

        boxOverlay.visibility = if (isOcr) View.GONE else View.VISIBLE
        if (isOcr) boxOverlay.clear() else boxOverlay.update(liveBoxes, liveImageW, liveImageH)

        // Guidance card
        val sev = severityColor()
        guidanceTag.text = when {
            latestLiveRiskScore >= 85 -> "위험"
            latestLiveRiskScore >= 55 -> "주의"
            else -> "안내"
        }
        guidanceTag.setTextColor(sev)
        guidanceAccent.background = rounded(sev, 3)
        guidanceMsg.text = latestGuide
        guidanceCard.background = glass(C_GLASS, withAlpha(sev, 0.28f), 24)

        // OCR panel
        ocrStatusLbl.text = liveOcrStatus.message
        ocrSpinner.visibility = if (liveOcrStatus == LiveOcrStatus.READING) View.VISIBLE else View.GONE
        val hasResult = !latestDetectedText.isNullOrBlank()
        ocrDivider.visibility = if (hasResult) View.VISIBLE else View.GONE
        ocrResultLbl.visibility = if (hasResult) View.VISIBLE else View.GONE
        ocrResultLbl.text = latestDetectedText.orEmpty()
        ocrCard.contentDescription = if (hasResult) "${latestDetectedText}. 두 번 탭하면 다시 읽어요." else liveOcrStatus.message

        // Mode bar
        styleModeButton(btnLive, btnLiveIcon, btnLiveLabel, active = !isOcr)
        styleModeButton(btnOcr, btnOcrIcon, btnOcrLabel, active = isOcr)
    }

    private fun styleModeButton(btn: LinearLayout, icon: IconView, label: TextView, active: Boolean) {
        btn.background = rounded(if (active) C_PRIMARY else C_PILL, 16)
        btn.elevation = if (active) dp(7).toFloat() else 0f
        btn.isSelected = active
        val tint = if (active) C_DARK else 0x99FFFFFF.toInt()
        icon.setColor(tint)
        label.setTextColor(tint)
    }

    private fun severityColor(): Int = when {
        latestLiveRiskScore >= 85 -> C_DANGER
        latestLiveRiskScore >= 55 -> C_WARNING
        else -> C_PRIMARY
    }

    // ── Mode switching ────────────────────────────────────────
    private fun toggleMode() {
        switchMode(if (currentMode == ProcessingMode.LIVE) ProcessingMode.TEXT else ProcessingMode.LIVE)
    }

    /** A mode change the user asked for (button or double tap): confirm it with a vibration. */
    private fun switchMode(mode: ProcessingMode) {
        haptics.play(HapticEvent.MODE_CHANGED)
        setMode(mode)
    }

    private fun setMode(mode: ProcessingMode) {
        currentMode = mode
        // Text mode turns it back on if the page is still dark.
        setTorch(false)
        speech.clearPendingGuidance()
        cameraExecutor.execute {
            sceneAnalyzer.reset()
            stabilityTracker.reset()
            pendingDetail = null
            darkFrameCount = 0
            lastLiveGuideMs = SystemClock.elapsedRealtime()
            displayedRiskScore = 0
        }

        latestDetectedText = null
        liveOcrStatus = LiveOcrStatus.SEARCHING
        latestLiveRiskScore = 0
        liveBoxes = emptyList()
        val message = when (mode) {
            ProcessingMode.LIVE -> {
                isLiveAnalysisRunning = false
                haptics.stopRepeatingPulses()
                "실시간 보행 안내를 시작합니다." + modeSwitchHint("문자 읽기")
            }
            ProcessingMode.TEXT -> {
                haptics.updateOcrPulseState(OcrHapticPulseState.SEARCHING)
                "문자 읽기 모드입니다. 카메라를 가까운 문자에 맞춰주세요." +
                    modeSwitchHint("실시간 안내", repeatHint = "화면을 한 번 누르면 읽은 문자를 다시 읽습니다.")
            }
        }

        // Always announce the new mode: users who can't see the screen need to hear which one is on.
        updateResponse(message, detectedText = null, shouldSpeak = true)
    }

    /**
     * Explains the switch gesture in the first few mode announcements, until it has likely
     * been learned. TalkBack users switch with the mode buttons, which TalkBack reads out.
     */
    private fun modeSwitchHint(target: String, repeatHint: String? = null): String {
        val accessibility = getSystemService(Context.ACCESSIBILITY_SERVICE) as AccessibilityManager
        if (accessibility.isTouchExplorationEnabled) return ""

        val preferences = getSharedPreferences(PREFERENCES, MODE_PRIVATE)
        val count = preferences.getInt(MODE_SWITCH_HINT_COUNT, 0)
        if (count >= 3) return ""

        preferences.edit().putInt(MODE_SWITCH_HINT_COUNT, count + 1).apply()
        return " 화면을 두 번 누르면 ${target}로 바뀝니다." + (repeatHint?.let { " $it" } ?: "")
    }

    // ── Camera ────────────────────────────────────────────────
    private fun startCameraWhenReady() {
        previewView.post { startCamera() }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            // 16:9 like the iPhone's 1920x1080 frames; 720p keeps small print readable.
            val resolution = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
                .build()
            val preview = Preview.Builder()
                .setResolutionSelector(resolution)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(cameraExecutor) { image -> analyzeFrame(image) } }
            provider.unbindAll()
            val viewPort = previewView.viewPort
            camera = if (viewPort != null) {
                val useCases = UseCaseGroup.Builder()
                    .setViewPort(viewPort)
                    .addUseCase(preview)
                    .addUseCase(analysis)
                    .build()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, useCases)
            } else {
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }
            isTorchOn = false
            setMode(currentMode)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun setTorch(on: Boolean) {
        val camera = camera ?: return
        if (!camera.cameraInfo.hasFlashUnit() || isTorchOn == on) return
        isTorchOn = on
        camera.cameraControl.enableTorch(on)
    }

    /** Runs on the camera thread; frames that arrive meanwhile are dropped. */
    private fun analyzeFrame(image: ImageProxy) {
        try {
            when (currentMode) {
                ProcessingMode.LIVE -> analyzeLiveFrame(image)
                ProcessingMode.TEXT -> analyzeTextFrame(image)
            }
        } catch (exception: Exception) {
            // Skip this frame; the next one retries.
            Log.w(TAG, "Frame analysis failed", exception)
        } finally {
            image.close()
        }
    }

    // ── Live guidance (on-device YOLO) ────────────────────────
    private fun analyzeLiveFrame(image: ImageProxy) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastLiveAnalysisMs < LIVE_ANALYSIS_INTERVAL_MS) return
        lastLiveAnalysisMs = now

        val detector = loadObjectDetectorIfNeeded() ?: return
        val frame = ImageProxyBitmapConverter.toUprightBitmap(image, cropToViewport = true)
        val width = frame.width
        val height = frame.height
        val pixels = IntArray(width * height)
        frame.getPixels(pixels, 0, width, 0, 0, width, height)

        val detections = detector.detect(pixels, width, height)
        if (currentMode != ProcessingMode.LIVE) return
        val result = sceneAnalyzer.analyze(detections, width.toDouble(), height.toDouble(), now / 1000.0)
        publishLiveResult(result, width, height, now)
    }

    private fun loadObjectDetectorIfNeeded(): ObjectDetector? {
        objectDetector?.let { return it }
        if (objectDetectorFailed) return null

        return try {
            val modelBytes = assets.open(ObjectDetector.MODEL_ASSET).use { it.readBytes() }
            ObjectDetector(modelBytes, threads = DETECTOR_THREADS).also { objectDetector = it }
        } catch (exception: Exception) {
            objectDetectorFailed = true
            Log.e(TAG, "Object detector unavailable", exception)
            updateResponse("비전 모델을 사용할 수 없습니다.", detectedText = null, shouldSpeak = true)
            null
        }
    }

    private fun publishLiveResult(result: LiveSceneResult, frameWidth: Int, frameHeight: Int, now: Long) {
        val hasNewGuide = result.voiceGuide.isNotEmpty()
        if (hasNewGuide) lastLiveGuideMs = now
        val shouldClearGuide = !hasNewGuide && now - lastLiveGuideMs > LIVE_GUIDE_DISPLAY_MS
        // Ranked by what is shown, so a capped object beside the path never hides one in it.
        val shown = result.detections.sortedByDescending { GuidanceRules.displayRisk(it) }
        val boxes = makeLiveBoxes(shown, frameWidth.toDouble(), frameHeight.toDouble())
        val riskScore = shown.firstOrNull()?.let { GuidanceRules.displayRisk(it) } ?: 0
        if (riskScore >= displayedRiskScore || now - displayedRiskMs > RISK_DISPLAY_HOLD_MS) {
            displayedRiskScore = riskScore
            displayedRiskMs = now
        }
        val displayedRisk = displayedRiskScore

        runOnUiThread {
            if (currentMode != ProcessingMode.LIVE) return@runOnUiThread
            isLiveAnalysisRunning = true
            if (hasNewGuide) latestGuide = result.voiceGuide
            else if (shouldClearGuide) latestGuide = ""
            latestLiveRiskScore = displayedRisk
            liveImageW = frameWidth.toFloat()
            liveImageH = frameHeight.toFloat()
            liveBoxes = boxes
            renderState()
        }

        if (hasNewGuide && currentMode == ProcessingMode.LIVE) {
            speech.speakGuidance(result.voiceGuide, result.voiceUrgency)
        }
    }

    /**
     * Only the most important object is highlighted, so low-vision users see one clear
     * target instead of a cluttered detector overlay. Boxes are normalized to the frame.
     */
    private fun makeLiveBoxes(detections: List<SceneDetection>, width: Double, height: Double): List<LiveGuidanceBox> {
        if (width <= 0 || height <= 0) return emptyList()
        return detections.take(LIVE_BOX_MAX_COUNT).map { detection ->
            LiveGuidanceBox(
                rect = RectRatio(
                    (detection.bbox.x1 / width).toFloat(),
                    (detection.bbox.y1 / height).toFloat(),
                    (detection.bbox.x2 / width).toFloat(),
                    (detection.bbox.y2 / height).toFloat()
                ),
                riskScore = GuidanceRules.displayRisk(detection),
                label = detection.koreanLabel,
                positionLabel = detection.position.koreanName
            )
        }
    }

    // ── Text reading (on-device ML Kit OCR) ───────────────────
    @OptIn(ExperimentalGetImage::class)
    private fun analyzeTextFrame(image: ImageProxy) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastTextAnalysisMs < TEXT_ANALYSIS_INTERVAL_MS) return
        val mediaImage = image.image ?: return
        lastTextAnalysisMs = now

        val rotation = image.imageInfo.rotationDegrees
        val sample = ImageProxyBitmapConverter.lumaSample(image)
        val movement = sample.movementScore(previousLuma)
        previousLuma = sample

        val text = Tasks.await(textRecognizer.process(InputImage.fromMediaImage(mediaImage, rotation)))
        val sideways = rotation % 180 != 0
        val lines = ocrLines(text, if (sideways) image.height else image.width, if (sideways) image.width else image.height)
        val region = OcrText.textRegion(lines)
        val analysis = OcrFrameAnalysis(
            textRegion = region,
            confidence = lines.filter { it.confidence >= OcrText.MINIMUM_REGION_CONFIDENCE }.maxOfOrNull { it.confidence } ?: 0f,
            blurScore = sample.blurScore(),
            movementScore = movement,
            brightness = sample.brightness,
            glareRatio = region?.let { sample.glareRatio(it, rotation) } ?: 0.0,
            timestampMs = now,
            readableLines = OcrText.readableLines(lines)
        )
        if (currentMode != ProcessingMode.TEXT) return

        pendingDetail?.let { (first, readAt) ->
            pendingDetail = null
            // A read left over from before a mode switch is stale.
            if (now - readAt < 1_000L) {
                val second = readDetail(image, analysis) ?: DetailRead.EMPTY
                speakDetail(first.confirmedBy(second))
                return
            }
        }

        val decision = stabilityTracker.update(analysis)
        updateLiveOcrStatus(decision.status)
        decision.hint?.let { speech.speakGuidance(it.message, RiskLevel.LOW) }
        lightUpIfDark(analysis)

        if (!decision.shouldRunFullOcr) return
        if (now - lastFullOcrMs < FULL_OCR_COOLDOWN_MS) {
            updateLiveOcrStatus(LiveOcrStatus.COOLING_DOWN)
            return
        }

        lastFullOcrMs = now
        haptics.stopRepeatingPulses()
        val detail = readDetail(image, analysis)
        if (detail != null && (detail.lines.isNotEmpty() || detail.droppedNumberLines > 0)) {
            pendingDetail = detail to now
        } else {
            readText(analysis.readableText)
        }
    }

    /** Recognized lines normalized to the upright frame (`width` x `height`). */
    private fun ocrLines(text: Text, width: Int, height: Int): List<OcrLine> {
        if (width <= 0 || height <= 0) return emptyList()
        return text.textBlocks.flatMap { it.lines }.mapNotNull { line ->
            val box = line.boundingBox ?: return@mapNotNull null
            OcrLine(
                text = line.text,
                box = RectRatio(
                    (box.left.toFloat() / width).coerceIn(0f, 1f),
                    (box.top.toFloat() / height).coerceIn(0f, 1f),
                    (box.right.toFloat() / width).coerceIn(0f, 1f),
                    (box.bottom.toFloat() / height).coerceIn(0f, 1f)
                ),
                confidence = line.confidence
            )
        }
    }

    /**
     * Reads the text region again from a crop around it, enlarged so small print gets more
     * of the recognizer's resolution. Null when there is no text region.
     */
    private fun readDetail(image: ImageProxy, analysis: OcrFrameAnalysis): DetailRead? {
        val region = analysis.textRegion ?: return null
        val frame = ImageProxyBitmapConverter.toUprightBitmap(image, cropToViewport = false)
        val left = ((region.left - DETAIL_MARGIN).coerceIn(0f, 1f) * frame.width).toInt()
        val top = ((region.top - DETAIL_MARGIN).coerceIn(0f, 1f) * frame.height).toInt()
        val right = ((region.right + DETAIL_MARGIN).coerceIn(0f, 1f) * frame.width).toInt()
        val bottom = ((region.bottom + DETAIL_MARGIN).coerceIn(0f, 1f) * frame.height).toInt()
        if (right - left < 8 || bottom - top < 8) return null

        var crop = Bitmap.createBitmap(frame, left, top, right - left, bottom - top)
        val scale = min(DETAIL_MAX_SCALE, DETAIL_TARGET_SIZE / max(crop.width, crop.height).toFloat())
        if (scale > 1.05f) {
            crop = Bitmap.createScaledBitmap(crop, (crop.width * scale).toInt(), (crop.height * scale).toInt(), true)
        }
        val text = Tasks.await(textRecognizer.process(InputImage.fromBitmap(crop, 0)))
        return OcrText.detailRead(ocrLines(text, crop.width, crop.height))
    }

    // shortcut: the torch stays on until the mode changes, since a lit page always looks bright.
    private fun lightUpIfDark(analysis: OcrFrameAnalysis) {
        darkFrameCount = if (analysis.brightness < DARK_BRIGHTNESS) darkFrameCount + 1 else 0
        if (darkFrameCount < 5 || isTorchOn || currentMode != ProcessingMode.TEXT) return
        if (camera?.cameraInfo?.hasFlashUnit() != true) return

        runOnUiThread { if (currentMode == ProcessingMode.TEXT) setTorch(true) }
        speech.speakGuidance("어두워서 불을 켰어요.", RiskLevel.LOW)
    }

    /** Reads the last text again, even though it was already spoken. */
    private fun repeatLastText() {
        if (currentMode != ProcessingMode.TEXT) return
        haptics.stopRepeatingPulses()
        speech.speak(latestDetectedText?.takeIf { it.isNotEmpty() } ?: "아직 읽은 문자가 없습니다.") {
            if (currentMode == ProcessingMode.TEXT) haptics.updateOcrPulseState(OcrHapticPulseState.SEARCHING)
        }
    }

    private fun speakDetail(read: DetailRead) {
        val notes = ArrayList<String>()
        if (read.droppedNumberLines > 0) {
            notes.add(
                if (read.lines.isEmpty()) "숫자가 있는 글자를 확실히 읽지 못했습니다. 휴대폰을 움직이지 말고 다시 비춰 주세요."
                else "숫자가 확실하지 않은 줄은 빼고 읽었습니다."
            )
        }
        if (read.missedLines > 0) notes.add("일부 글자는 읽지 못했습니다.")
        readText(read.lines.joinToString(" "), notes.joinToString(" "))
    }

    private fun readText(detectedText: String, notes: String = "") {
        // Only the text itself: the mode was announced when it started.
        val voiceGuide = if (detectedText.isEmpty() && notes.isEmpty()) {
            "현재 화면에서 읽을 수 있는 문자를 찾지 못했습니다."
        } else {
            listOf(detectedText, notes).filter { it.isNotEmpty() }.joinToString(" ")
        }
        val shouldSpeak = if (detectedText.isEmpty()) notes.isNotEmpty() else duplicateSuppressor.shouldSpeak(detectedText)

        if (shouldSpeak) {
            haptics.stopRepeatingPulses()
            haptics.play(HapticEvent.READABLE_TEXT_CONFIRMED)
        }
        updateResponse(voiceGuide, detectedText, shouldSpeak)

        runOnUiThread {
            if (currentMode != ProcessingMode.TEXT) return@runOnUiThread
            liveOcrStatus = LiveOcrStatus.COOLING_DOWN
            if (!shouldSpeak) haptics.updateOcrPulseState(OcrHapticPulseState.SEARCHING)
            renderState()
        }
    }

    private fun updateResponse(voiceGuide: String, detectedText: String?, shouldSpeak: Boolean) {
        runOnUiThread {
            latestGuide = voiceGuide
            latestDetectedText = detectedText
            renderState()
        }

        if (!shouldSpeak) return
        haptics.stopRepeatingPulses()
        speech.speak(voiceGuide) {
            if (currentMode == ProcessingMode.TEXT) haptics.updateOcrPulseState(OcrHapticPulseState.SEARCHING)
        }
    }

    private fun updateLiveOcrStatus(status: LiveOcrStatus) {
        runOnUiThread {
            if (currentMode != ProcessingMode.TEXT) return@runOnUiThread
            liveOcrStatus = status
            if (latestDetectedText.isNullOrEmpty()) latestGuide = status.message

            when (status) {
                LiveOcrStatus.SEARCHING, LiveOcrStatus.COOLING_DOWN -> haptics.updateOcrPulseState(OcrHapticPulseState.SEARCHING)
                LiveOcrStatus.DETECTED -> haptics.updateOcrPulseState(OcrHapticPulseState.DETECTED)
                LiveOcrStatus.STABILIZING -> haptics.updateOcrPulseState(OcrHapticPulseState.STABILIZING)
                LiveOcrStatus.READING, LiveOcrStatus.UNAVAILABLE -> haptics.stopRepeatingPulses()
            }
            renderState()
        }
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    // ── Layout / drawable helpers ─────────────────────────────
    private fun dp(v: Int) = (v * density).toInt()

    private fun matchParent() = FrameLayout.LayoutParams(MATCH, MATCH)

    private fun rowLp(top: Int, h: Int) = LinearLayout.LayoutParams(MATCH, dp(h)).apply {
        topMargin = dp(top); marginStart = dp(24); marginEnd = dp(24)
    }

    private fun cardLp() = LinearLayout.LayoutParams(MATCH, WRAP).apply {
        marginStart = dp(24); marginEnd = dp(24)
    }

    private fun modeBarLp() = LinearLayout.LayoutParams(MATCH, WRAP).apply {
        topMargin = dp(12); bottomMargin = dp(30); marginStart = dp(18); marginEnd = dp(18)
    }

    private fun rounded(color: Int, r: Int) = GradientDrawable().apply {
        setColor(color); cornerRadius = dp(r).toFloat()
    }

    private fun circle(color: Int) = GradientDrawable().apply {
        shape = GradientDrawable.OVAL; setColor(color)
    }

    private fun capsule(fill: Int, stroke: Int, r: Int) = GradientDrawable().apply {
        setColor(fill); cornerRadius = dp(r).toFloat(); setStroke(dpStroke(1f), stroke)
    }

    private fun glass(fill: Int, stroke: Int, r: Int) = GradientDrawable().apply {
        setColor(fill); cornerRadius = dp(r).toFloat(); setStroke(dpStroke(1.5f), stroke)
    }

    private fun dpStroke(v: Float) = (v * density).toInt().coerceAtLeast(1)

    private fun spinner(scale: Float) = ProgressBar(this).apply {
        isIndeterminate = true
        visibility = View.GONE
        scaleX = scale; scaleY = scale
        indeterminateTintList = android.content.res.ColorStateList.valueOf(C_PRIMARY)
    }

    private fun withAlpha(color: Int, fraction: Float): Int {
        val a = (fraction.coerceIn(0f, 1f) * 255).toInt()
        return (color and 0x00FFFFFF) or (a shl 24)
    }

    companion object {
        private const val TAG = "Gilnun"
        private const val CAMERA_REQUEST = 10
        private const val PREFERENCES = "gilnun"
        private const val MODE_SWITCH_HINT_COUNT = "modeSwitchHintCount"

        /** About five detector runs per second at most: fast enough to track approaching objects. */
        private const val LIVE_ANALYSIS_INTERVAL_MS = 200L
        private const val DETECTOR_THREADS = 4

        /** How long a spoken live message stays on the guidance card. */
        private const val LIVE_GUIDE_DISPLAY_MS = 3_000L

        /**
         * A higher risk stays on screen this long after it was last seen, so the card color
         * doesn't flicker when a score hovers around a threshold.
         */
        private const val RISK_DISPLAY_HOLD_MS = 1_500L
        private const val LIVE_BOX_MAX_COUNT = 1

        private const val TEXT_ANALYSIS_INTERVAL_MS = 160L
        private const val FULL_OCR_COOLDOWN_MS = 3_000L

        /** Frames this dark mean auto exposure has run out, so text mode lights the page. */
        private const val DARK_BRIGHTNESS = 45.0
        private const val DETAIL_MARGIN = 0.05f
        private const val DETAIL_TARGET_SIZE = 1600f
        private const val DETAIL_MAX_SCALE = 2.5f

        // Palette (mirrors iOS `P`).
        private val C_PRIMARY = 0xFF45B8FF.toInt()
        private val C_LIVE = 0xFF29DE8F.toInt()
        private val C_WARNING = 0xFFFF941F.toInt()
        private val C_DANGER = 0xFFFF3B3B.toInt()
        private val C_GLASS = 0xB8172138.toInt()
        private val C_GLASS_STROKE = 0x1AFFFFFF
        private val C_DIM_TEXT = 0x73FFFFFF
        private val C_PILL = 0x12FFFFFF
        private val C_DARK = 0xFF050F24.toInt()

        private const val MATCH = ViewGroup.LayoutParams.MATCH_PARENT
        private const val WRAP = ViewGroup.LayoutParams.WRAP_CONTENT
    }
}
