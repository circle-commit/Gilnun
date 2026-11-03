package com.kojingapla.glass

import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator

enum class HapticEvent {
    READABLE_TEXT_CONFIRMED,
    MODE_CHANGED,
    /** The walk signal just turned green: one long, steady vibration. */
    SIGNAL_GO,
    /** Red, flashing or uncertain signal: three short taps, "wait". */
    SIGNAL_WAIT,
    /** Something in the way right now: two strong pulses, felt even in traffic noise. */
    DANGER
}

enum class OcrHapticPulseState {
    SEARCHING,
    DETECTED,
    STABILIZING
}

/**
 * Vibration cues for text reading and mode changes. Port of iOS `HapticFeedbackManager.swift`:
 * a slow single pulse while searching for text, a quicker double pulse once text is found,
 * slowing down when nothing is read for a while.
 */
class HapticFeedbackManager(context: Context) {
    @Suppress("DEPRECATION")
    private val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    private val main = Handler(Looper.getMainLooper())
    private var pulseState: OcrHapticPulseState? = null
    private var pulseStartedAtMs = 0L
    private val pulse = Runnable { emitPulseAndReschedule() }

    fun updateOcrPulseState(state: OcrHapticPulseState) {
        main.post {
            if (pulseState == state) return@post
            pulseState = state
            pulseStartedAtMs = SystemClock.elapsedRealtime()
            emitPulse(state)
            scheduleNextPulse(interval(state))
        }
    }

    fun stopRepeatingPulses() {
        main.post {
            main.removeCallbacks(pulse)
            pulseState = null
        }
    }

    fun play(event: HapticEvent) {
        main.post {
            when (event) {
                HapticEvent.READABLE_TEXT_CONFIRMED -> vibrate(doublePulse())
                // A sharp tick that feels different from the text search pulses.
                HapticEvent.MODE_CHANGED -> vibrate(
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        VibrationEffect.createPredefined(VibrationEffect.EFFECT_HEAVY_CLICK)
                    } else {
                        VibrationEffect.createOneShot(20, VibrationEffect.DEFAULT_AMPLITUDE)
                    }
                )
                // Pedestrian-signal aids pair speech with vibration for loud streets (Oko, LYTNet's app).
                HapticEvent.SIGNAL_GO -> vibrate(VibrationEffect.createOneShot(600, VibrationEffect.DEFAULT_AMPLITUDE))
                HapticEvent.SIGNAL_WAIT -> vibrate(VibrationEffect.createWaveform(longArrayOf(0, 60, 120, 60, 120, 60), -1))
                HapticEvent.DANGER -> vibrate(strongDoublePulse())
            }
        }
    }

    private fun scheduleNextPulse(intervalMs: Long) {
        main.removeCallbacks(pulse)
        main.postDelayed(pulse, intervalMs)
    }

    private fun emitPulseAndReschedule() {
        val state = pulseState ?: return
        emitPulse(state)
        scheduleNextPulse(interval(state))
    }

    private fun emitPulse(state: OcrHapticPulseState) {
        when (state) {
            OcrHapticPulseState.SEARCHING -> vibrate(VibrationEffect.createOneShot(30, VibrationEffect.DEFAULT_AMPLITUDE))
            OcrHapticPulseState.DETECTED, OcrHapticPulseState.STABILIZING -> vibrate(doublePulse())
        }
    }

    private fun doublePulse(): VibrationEffect = VibrationEffect.createWaveform(longArrayOf(0, 35, 80, 35), -1)

    private fun strongDoublePulse(): VibrationEffect {
        val timings = longArrayOf(0, 150, 100, 150)
        return if (vibrator?.hasAmplitudeControl() == true) {
            VibrationEffect.createWaveform(timings, intArrayOf(0, 255, 0, 255), -1)
        } else {
            VibrationEffect.createWaveform(timings, -1)
        }
    }

    private fun vibrate(effect: VibrationEffect) {
        val vibrator = vibrator ?: return
        if (vibrator.hasVibrator()) vibrator.vibrate(effect)
    }

    private fun interval(state: OcrHapticPulseState): Long {
        if (SystemClock.elapsedRealtime() - pulseStartedAtMs >= TIMEOUT_MS) return INTERVAL_AFTER_TIMEOUT_MS
        return when (state) {
            OcrHapticPulseState.SEARCHING -> SEARCHING_INTERVAL_MS
            OcrHapticPulseState.DETECTED, OcrHapticPulseState.STABILIZING -> CANDIDATE_INTERVAL_MS
        }
    }

    private companion object {
        const val SEARCHING_INTERVAL_MS = 900L
        const val CANDIDATE_INTERVAL_MS = 450L
        const val TIMEOUT_MS = 12_000L
        const val INTERVAL_AFTER_TIMEOUT_MS = 1_500L
    }
}
