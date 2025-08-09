package com.specknet.pdiotapp

import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.specknet.pdiotapp.live.ViewActivity
import com.specknet.pdiotapp.utils.RESpeckLiveData
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.system.measureTimeMillis

@RunWith(AndroidJUnit4::class)
class MyPageTest {

//    @Test
//    /**
//     * To perform this test, modify the prediction function in ViewActivity to return the number of dataPoints
//     */
//    fun checkPredictPageRefresh() {
//        // Launch the main activity containing the page to be tested
//        ActivityScenario.launch(ViewActivity::class.java)
//
//        val context = InstrumentationRegistry.getInstrumentation().targetContext
//        // Click on the view activity button to start test
////        Espresso.onView(withId(R.id.view_activity)).perform(click())
//        // Mock data to be send to the Respeck broadcast Reciever
//        for (index in 0 until 50) {
////            if (index % 25 == 0 && index > 0) {
////                Espresso.onView(withId(R.id.activity))
////                    .check(matches(withText((6 * index).toString())))
////            }
//            val mockIntention = Intent("com.specknet.respeck.RESPECK_LIVE_BROADCAST")
//            val respeckLiveData = RESpeckLiveData(0, 0, 0, index.toFloat(), 0f, 0f, 0f, 0, false)
//            mockIntention.putExtra("respeck_live_data", respeckLiveData)
//            context.sendBroadcast(mockIntention)
//            // Mock 25Hz frequency
//            Thread.sleep(40)
//            // Once 2 seconds worth of data is sent check if the prediction text view is updated
//        }
//    }


//    @Test
//    /**
//     * Identify that the number of datapoints stored never increased beyond 12 broadcasts
//     */
//    fun checkFunctionAsync() {
//        // Launch the main activity containing the page to be tested
//        ActivityScenario.launch(ViewActivity::class.java)
//        val context = InstrumentationRegistry.getInstrumentation().targetContext
//        // Click on the view activity button to start test
////        Espresso.onView(withId(R.id.view_activity)).perform(click())
//        // Mock data to be send to the Respeck broadcast Reciever
//        for (index in 0 until 2000) {
//            val mockIntention = Intent("com.specknet.respeck.RESPECK_LIVE_BROADCAST")
//            val respeckLiveData = RESpeckLiveData(0, 0, 0, index.toFloat(), 0f, 0f, 0f, 0, false)
//            mockIntention.putExtra("respeck_live_data", respeckLiveData)
//            context.sendBroadcast(mockIntention)
//            // Mock 25Hz frequency
//            Thread.sleep(40)
//            val test = { viewActivity: ViewActivity -> viewActivity.getRespeckData() };
//            test.toString()
//            // Once 2 seconds worth of data is sent check if the prediction text view is updated
//        }
//    }
}

