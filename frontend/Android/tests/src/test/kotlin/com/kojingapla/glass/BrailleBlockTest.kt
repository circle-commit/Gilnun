package com.kojingapla.glass

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Port of iOS `runBrailleTrackerTests` (BrailleTests.swift). */
class BrailleBlockTest {
    /** Blocks filling the given columns in the given rows. */
    private fun mask(columns: IntRange, rows: IntRange = 40 until 80): BrailleMask {
        val cells = BooleanArray(BrailleMask.COLUMNS * BrailleMask.ROWS)
        for (row in rows) for (column in columns) cells[row * BrailleMask.COLUMNS + column] = true
        return BrailleMask(cells)
    }

    private val left = mask(2 until 8)
    private val center = mask(19 until 26)

    @Test
    fun announcesOnceAfterThreeFrames() {
        val tracker = BrailleBlockTracker()
        assertNull(tracker.update(left, 0.0, true), "spoke after one frame")
        assertNull(tracker.update(left, 0.2, true), "spoke after two frames")
        assertEquals("왼쪽에 점자블록이 있어요.", tracker.update(left, 0.4, true), "left not announced")
        assertNull(tracker.update(left, 0.6, true), "repeated itself")

        // A new side waits for the side-change interval.
        tracker.update(center, 1.0, true)
        tracker.update(center, 1.2, true)
        assertNull(tracker.update(center, 1.4, true), "side change too soon")
        assertEquals("앞에 점자블록이 있어요.", tracker.update(center, 4.4, true), "center not announced")

        // Gone for two seconds, then back: announced again.
        for (frame in 0 until BrailleBlockTracker.FRAMES_TO_LOSE) {
            assertNull(tracker.update(BrailleMask.EMPTY, 5 + frame * 0.2, true), "spoke without blocks")
        }
        for (frame in 0 until 3) {
            val message = tracker.update(center, 8 + frame * 0.2, true)
            assertEquals(frame == 2, message != null, "return after loss, frame $frame -> $message")
        }
    }

    @Test
    fun urgentSpeechDefersTheAnnouncement() {
        val tracker = BrailleBlockTracker()
        for (frame in 0 until 4) assertNull(tracker.update(left, frame * 0.2, false), "spoke over urgent guidance")
        assertEquals("왼쪽에 점자블록이 있어요.", tracker.update(left, 1.0, true), "deferred announcement lost")
    }

    @Test
    fun flickerNeverConfirms() {
        val tracker = BrailleBlockTracker()
        for (frame in 0 until 20) {
            assertNull(tracker.update(if (frame % 2 == 0) center else BrailleMask.EMPTY, frame * 0.2, true), "flicker announced")
        }
    }

    @Test
    fun onlyTheWalkingAreaCounts() {
        assertNull(BrailleBlockTracker.position(mask(0 until 45, 0 until 30)), "far blocks counted")
        assertNull(BrailleBlockTracker.position(mask(40 until 41, 70 until 80)), "speck counted")
        assertEquals(HorizontalPosition.RIGHT, BrailleBlockTracker.position(mask(38 until 44)), "right side")
        val centerColumn = BrailleBlockTracker.centerColumn(left)!!
        assertTrue(centerColumn < BrailleMask.COLUMNS / 3.0, "left blocks center on the left: $centerColumn")
    }
}
