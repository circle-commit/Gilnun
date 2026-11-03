package com.kojingapla.glass

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import java.util.Locale

/**
 * Text-to-speech with live-guidance priorities. Port of iOS `SpeechManager.swift`.
 *
 * Mode announcements and text reading cut off whatever is being said; live guidance
 * waits for a sentence of equal or higher urgency, keeping only the newest one.
 * Speaks as navigation guidance and lowers other apps' audio while talking.
 */
class SpeechManager(context: Context) : TextToSpeech.OnInitListener {
    private val main = Handler(Looper.getMainLooper())
    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private val attributes = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()
    private val focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
        .setAudioAttributes(attributes)
        .build()
    private val tts = TextToSpeech(context.applicationContext, this)

    // Main thread only.
    private var isReady = false
    private var waitingForEngine: (() -> Unit)? = null
    private var currentUtteranceId: String? = null
    /** Urgency of what is being spoken now; null when idle. */
    private var currentUrgency: RiskLevel? = null
    /** Newest live guidance waiting for the current sentence to finish. */
    private var pendingGuidance: PendingGuidance? = null
    private val finishHandlers = HashMap<String, () -> Unit>()
    private var nextUtteranceNumber = 0L
    private var currentLanguage: String? = null

    private class PendingGuidance(val text: String, val urgency: RiskLevel, val createdAtMs: Long)

    override fun onInit(status: Int) {
        main.post {
            if (status != TextToSpeech.SUCCESS) return@post
            tts.setAudioAttributes(attributes)
            tts.setSpeechRate(SPEECH_RATE)
            tts.setPitch(SPEECH_PITCH)
            tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String) = Unit
                override fun onDone(utteranceId: String) {
                    main.post { utteranceEnded(utteranceId, finished = true) }
                }

                @Deprecated("Deprecated in Java")
                override fun onError(utteranceId: String) {
                    main.post { utteranceEnded(utteranceId, finished = false) }
                }

                override fun onStop(utteranceId: String, interrupted: Boolean) {
                    main.post { utteranceEnded(utteranceId, finished = false) }
                }
            })
            isReady = true
            waitingForEngine?.invoke()
            waitingForEngine = null
        }
    }

    /** Speaks right away, cutting off anything in progress (mode announcements, text reading). */
    fun speak(text: String, onFinish: (() -> Unit)? = null) {
        if (text.isEmpty()) return
        main.post {
            pendingGuidance = null
            // Only a critical live warning may cut this off.
            startSpeaking(text, RiskLevel.HIGH, onFinish)
        }
    }

    /**
     * Speaks live guidance without cutting off a sentence of equal or higher urgency.
     * Less urgent guidance waits for the current sentence; only the newest one is kept.
     */
    fun speakGuidance(text: String, urgency: RiskLevel) {
        if (text.isEmpty()) return
        main.post {
            val speakingUrgency = currentUrgency
            if (currentUtteranceId != null && speakingUrgency != null && urgency <= speakingUrgency) {
                pendingGuidance = PendingGuidance(text, urgency, SystemClock.elapsedRealtime())
                return@post
            }
            pendingGuidance = null
            startSpeaking(text, urgency, null)
        }
    }

    /** Drops live guidance that is still waiting to be spoken (e.g. after a mode switch). */
    fun clearPendingGuidance() {
        main.post { pendingGuidance = null }
    }

    fun stop() {
        main.post {
            pendingGuidance = null
            tts.stop()
        }
    }

    fun shutdown() {
        main.removeCallbacksAndMessages(null)
        tts.stop()
        tts.shutdown()
        audioManager.abandonAudioFocusRequest(focusRequest)
    }

    private fun startSpeaking(text: String, urgency: RiskLevel, onFinish: (() -> Unit)?) {
        if (!isReady) {
            // The engine is still starting; keep only the latest request.
            waitingForEngine = { startSpeaking(text, urgency, onFinish) }
            return
        }

        val utteranceId = "gilnun-${nextUtteranceNumber++}"
        if (onFinish != null) finishHandlers[utteranceId] = onFinish
        currentUtteranceId = utteranceId
        currentUrgency = urgency

        useLanguage(if (text.any { it in '가'..'힯' }) Locale.KOREAN else Locale.US)
        audioManager.requestAudioFocus(focusRequest)
        val result = tts.speak(text, TextToSpeech.QUEUE_FLUSH, Bundle(), utteranceId)
        if (result != TextToSpeech.SUCCESS) utteranceEnded(utteranceId, finished = false)
    }

    private fun utteranceEnded(utteranceId: String, finished: Boolean) {
        val handler = finishHandlers.remove(utteranceId)
        if (utteranceId == currentUtteranceId) {
            currentUtteranceId = null
            currentUrgency = null
        }
        handler?.invoke()
        if (finished) speakPendingGuidance()
        if (currentUtteranceId == null) audioManager.abandonAudioFocusRequest(focusRequest)
    }

    private fun speakPendingGuidance() {
        val pending = pendingGuidance ?: return
        if (currentUtteranceId != null) return
        pendingGuidance = null
        // Waiting guidance older than this is dropped; the scene has likely changed.
        if (SystemClock.elapsedRealtime() - pending.createdAtMs > PENDING_GUIDANCE_LIFETIME_MS) return
        startSpeaking(pending.text, pending.urgency, null)
    }

    /** Prefers the best installed voice for the language; network voices would send text off the phone. */
    private fun useLanguage(locale: Locale) {
        if (currentLanguage == locale.language) return
        currentLanguage = locale.language
        tts.language = locale
        tts.voices
            ?.filter { it.locale.language == locale.language && !it.isNetworkConnectionRequired }
            ?.filterNot { it.features?.contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED) == true }
            ?.sortedWith(compareByDescending<Voice> { it.quality }.thenBy { it.latency })
            ?.firstOrNull()
            ?.let { tts.voice = it }
    }

    private companion object {
        const val SPEECH_RATE = 0.92f
        const val SPEECH_PITCH = 1.04f
        const val PENDING_GUIDANCE_LIFETIME_MS = 2_000L
    }
}
