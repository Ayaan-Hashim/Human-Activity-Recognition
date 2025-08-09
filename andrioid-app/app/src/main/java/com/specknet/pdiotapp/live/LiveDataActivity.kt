package com.specknet.pdiotapp.live

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet
import com.specknet.pdiotapp.R
import com.specknet.pdiotapp.utils.Constants
import com.specknet.pdiotapp.utils.RESpeckLiveData
import kotlin.collections.ArrayList


class LiveDataActivity : AppCompatActivity() {

    // global graph variables
    private lateinit var datasetResAccelX: LineDataSet
    private lateinit var datasetResAccelY: LineDataSet
    private lateinit var datasetResAccelZ: LineDataSet

    private lateinit var datasetResGyroX: LineDataSet
    private lateinit var datasetResGyroY: LineDataSet
    private lateinit var datasetResGyroZ: LineDataSet

    private lateinit var allRespeckAccelData: LineData

    private lateinit var allRespeckGyroData: LineData

    private lateinit var respeckAccelChart: LineChart
    private lateinit var respeckGyroChart: LineChart

    private lateinit var respeckLiveUpdateReceiver: BroadcastReceiver
    private lateinit var looperRespeck: Looper

    private val filterTestRespeck = IntentFilter(Constants.ACTION_RESPECK_LIVE_BROADCAST)
    private var time = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_live_data)

        setupCharts()

        // set up the broadcast receiver
        respeckLiveUpdateReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {

                Log.i("thread", "I am running on thread = " + Thread.currentThread().name)

                val action = intent.action

                if (action == Constants.ACTION_RESPECK_LIVE_BROADCAST) {

                    val liveData =
                        intent.getSerializableExtra(Constants.RESPECK_LIVE_DATA) as RESpeckLiveData
                    Log.d("Live", "onReceive: liveData = $liveData")

                    // get all relevant intent contents
                    val accelX = liveData.accelX
                    val accelY = liveData.accelY
                    val accelZ = liveData.accelZ
                    val gyroX = liveData.gyro.x
                    val gyroY = liveData.gyro.y
                    val gyroZ = liveData.gyro.z

                    time += 1
                    updateGraph("respeckAccel", accelX, accelY, accelZ)
                    updateGraph("respeckGyro", gyroX, gyroY, gyroZ)

                }
            }
        }

        // register receiver on another thread
        val handlerThreadRespeck = HandlerThread("bgThreadRespeckLive")
        handlerThreadRespeck.start()
        looperRespeck = handlerThreadRespeck.looper
        val handlerRespeck = Handler(looperRespeck)
        this.registerReceiver(respeckLiveUpdateReceiver, filterTestRespeck, null, handlerRespeck)
    }


    private fun setupCharts() {
        respeckAccelChart = findViewById(R.id.respeck_accel_chart)
        respeckGyroChart = findViewById(R.id.respeck_gyro_chart)

        // Respeck Acceleration
        time = 0f
        val entriesResAccelX = ArrayList<Entry>()
        val entriesResAccelY = ArrayList<Entry>()
        val entriesResAccelZ = ArrayList<Entry>()

        datasetResAccelX = LineDataSet(entriesResAccelX, "Accel X")
        datasetResAccelY = LineDataSet(entriesResAccelY, "Accel Y")
        datasetResAccelZ = LineDataSet(entriesResAccelZ, "Accel Z")

        datasetResAccelX.setDrawCircles(false)
        datasetResAccelY.setDrawCircles(false)
        datasetResAccelZ.setDrawCircles(false)

        datasetResAccelX.color = ContextCompat.getColor(
            this, R.color.red
        )
        datasetResAccelY.color =
            ContextCompat.getColor(
                this, R.color.green
            )
        datasetResAccelZ.color =
            ContextCompat.getColor(
                this, R.color.blue
            )

        val dataSetsRes = ArrayList<ILineDataSet>()
        dataSetsRes.add(datasetResAccelX)
        dataSetsRes.add(datasetResAccelY)
        dataSetsRes.add(datasetResAccelZ)

        allRespeckAccelData = LineData(dataSetsRes)
        respeckAccelChart.data = allRespeckAccelData
        respeckAccelChart.description.text = String.format("Live acceleration data from connected RESpeck")
        respeckAccelChart.invalidate()

        // Th

        time = 0f
        val entriesResGyroX = ArrayList<Entry>()
        val entriesResGyroY = ArrayList<Entry>()
        val entriesResGyroZ = ArrayList<Entry>()

        datasetResGyroX = LineDataSet(entriesResGyroX, "Gyro X")
        datasetResGyroY = LineDataSet(entriesResGyroY, "Gyro Y")
        datasetResGyroZ = LineDataSet(entriesResGyroZ, "Gyro Z")

        datasetResGyroX.setDrawCircles(false)
        datasetResGyroY.setDrawCircles(false)
        datasetResGyroZ.setDrawCircles(false)

        datasetResGyroX.color = ContextCompat.getColor(
            this, R.color.red
        )
        datasetResGyroY.color = ContextCompat.getColor(
            this, R.color.green
        )
        datasetResGyroZ.color = ContextCompat.getColor(
            this, R.color.blue
        )

        val dataSetsThingy = ArrayList<ILineDataSet>()
        dataSetsThingy.add(datasetResGyroX)
        dataSetsThingy.add(datasetResGyroY)
        dataSetsThingy.add(datasetResGyroZ)

        allRespeckGyroData = LineData(dataSetsThingy)
        respeckGyroChart.data = allRespeckGyroData
        respeckGyroChart.description.text = String.format("Live gyroscope data from connected RESpeck")
        respeckGyroChart.invalidate()
    }

    fun updateGraph(graph: String, x: Float, y: Float, z: Float) {
        // take the first element from the queue
        // and update the graph with it
        if (graph == "respeckAccel") {
            datasetResAccelX.addEntry(Entry(time, x))
            datasetResAccelY.addEntry(Entry(time, y))
            datasetResAccelZ.addEntry(Entry(time, z))

            runOnUiThread {
                allRespeckAccelData.notifyDataChanged()
                respeckAccelChart.notifyDataSetChanged()
                respeckAccelChart.invalidate()
                respeckAccelChart.setVisibleXRangeMaximum(150f)
                respeckAccelChart.moveViewToX(respeckAccelChart.lowestVisibleX + 40)
            }
        } else if (graph == "respeckGyro") {
            datasetResGyroX.addEntry(Entry(time, x))
            datasetResGyroY.addEntry(Entry(time, y))
            datasetResGyroZ.addEntry(Entry(time, z))

            runOnUiThread {
                allRespeckGyroData.notifyDataChanged()
                respeckGyroChart.notifyDataSetChanged()
                respeckGyroChart.invalidate()
                respeckGyroChart.setVisibleXRangeMaximum(150f)
                respeckGyroChart.moveViewToX(respeckGyroChart.lowestVisibleX + 40)
            }
        }


    }


    override fun onDestroy() {
        super.onDestroy()
        unregisterReceiver(respeckLiveUpdateReceiver)
        looperRespeck.quit()
    }
}