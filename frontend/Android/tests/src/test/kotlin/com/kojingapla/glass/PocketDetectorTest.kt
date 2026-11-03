package com.kojingapla.glass

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class PocketDetectorTest {
    @Test
    fun pausesOnlyWhenPutAwayForAWhile() {
        val pocket = PocketDetector()
        assertNull(pocket.update(true, 10.0), "covering starts")
        assertNull(pocket.update(false, 10.4), "a hand passing over the sensor changes nothing")
        assertNull(pocket.update(true, 20.0), "covered again")
        assertNull(pocket.update(true, 21.0), "not long enough yet")
        assertEquals(PocketDetector.Change.PUT_AWAY, pocket.update(true, 21.6), "covered for 1.5 s is a pocket")
        assertTrue(pocket.isPutAway)
        assertNull(pocket.update(true, 30.0), "said once")
        assertNull(pocket.update(false, 31.0), "uncovered starts")
        assertEquals(PocketDetector.Change.TAKEN_OUT, pocket.update(false, 31.6), "out for half a second resumes")
        assertNull(pocket.update(false, 40.0), "said once")
    }
}
