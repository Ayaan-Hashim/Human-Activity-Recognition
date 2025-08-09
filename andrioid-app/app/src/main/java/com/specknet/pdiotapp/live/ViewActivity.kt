package com.specknet.pdiotapp.live

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.res.AssetFileDescriptor
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.content.res.AppCompatResources
import androidx.lifecycle.lifecycleScope
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import com.specknet.pdiotapp.R
import com.specknet.pdiotapp.utils.Constants
import com.specknet.pdiotapp.utils.RESpeckLiveData
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.lang.Thread.currentThread
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Live prediction page of app
 */
class ViewActivity : AppCompatActivity() {

    var time = 0f
    private val predictionActions: ArrayList<String> = arrayListOf(
        "ascending stairs",
        "descending stairs",
        "lying down back",
        "lying down on left",
        "lying down on stomach",
        "lying down right",
        "miscellaneous movements",
        "normal walking",
        "running",
        "shuffle walking",
        "sitting"
    )
    private val dynamicActivities: ArrayList<String> = arrayListOf(
        "ascending stairs",
        "descending stairs",
        "miscellaneous movements",
        "normal walking",
        "running",
        "shuffle walking"
    )
    private val breathingAction: ArrayList<String> =
        arrayListOf("normal", "coughing", "hyperventilating", "other")

    // global broadcast receiver so we can unregister it
    private lateinit var respeckLiveUpdateReceiver: BroadcastReceiver
    private lateinit var looperRespeck: Looper
    private lateinit var predictedActivity: TextView
    private lateinit var predictedBreathing: TextView
    private lateinit var respeckConnectionStatus: TextView
    private lateinit var visualRep: ImageView
    private lateinit var firstModel: Interpreter
    private lateinit var secondModel: Interpreter
    private lateinit var handlerThreadRespeck : HandlerThread
    private var job: Job? = null
    private var prevActivity: String = ""
    private var prevBreathing: String = ""
    private var activityPredModel: String =
        "Models/final-model-lvl1.tflite"
    private var breathingPredModel: String = "Models/final-model-level2.tflite"
    private val respeckDataBuffer = mutableListOf<Float>()
    private lateinit var firestoreDb: FirebaseFirestore
    private val filterTestRespeck = IntentFilter(Constants.ACTION_RESPECK_LIVE_BROADCAST)
    private var lastSavedActivity: String? = null
    private var lastSavedBreathing: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.view_activity)

        // Initialize views to change during predictions
        this@ViewActivity.predictedActivity = findViewById(R.id.activity)
        this@ViewActivity.predictedBreathing = findViewById(R.id.breathing)
        this@ViewActivity.respeckConnectionStatus = findViewById(R.id.respeckConnection)
        this@ViewActivity.visualRep = findViewById(R.id.activityVisualRep)
        // Initialize the ML models along with the database
        firstModel =
            Interpreter(loadModel(activityPredModel))
        secondModel = Interpreter(loadModel(breathingPredModel))
        firestoreDb = FirebaseFirestore.getInstance()
        // set up the broadcast receiver
        respeckLiveUpdateReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {

                Log.i("thread", "I am running on thread = " + currentThread().name)

                val action = intent.action

                if (action == Constants.ACTION_RESPECK_LIVE_BROADCAST) {
                    runOnUiThread {
                        respeckConnectionStatus.text = Constants.ON
                    }
                    val liveData =
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                            intent.getSerializableExtra(
                                Constants.RESPECK_LIVE_DATA,
                                RESpeckLiveData::class.java
                            )!!
                        } else {
                            intent.getSerializableExtra(Constants.RESPECK_LIVE_DATA) as RESpeckLiveData
                        }
                    Log.d("Live", "onReceive: liveData = $liveData")
                    // Add live data to stream
                    respeckDataBuffer.addLiveData(liveData)
                    if (respeckDataBuffer.size >= 300) {
                        // Once buffer reaches size, apply prediction
                        val data = respeckDataBuffer.toFloatArray().copyOfRange(0, 300)
                        if (job == null || job!!.isCompleted) {
                            job =
                                    lifecycleScope.launch {
                                        this@ViewActivity.applyPrediction(context, data)
//t
                                    }
                        }
                    }
                        if (respeckDataBuffer.size > 300) {
                            respeckDataBuffer.subList(0, 150).clear()
                        }
                    time += 1
                }
            }
        }
        // register receiver on another thread
        handlerThreadRespeck = HandlerThread("bgLiveRespeckDataCollect")
        handlerThreadRespeck.start()
        looperRespeck = handlerThreadRespeck.looper
        val handlerRespeck = Handler(looperRespeck)
        registerReciever(handlerRespeck)
    }

    /**
     * Applied prediction to UI page
     * @param context
     * @param data Input to Ml model
     */
    private fun applyPrediction(context: Context, data: FloatArray) {
        val currentActivity: String = predict(
            firstModel,
            data,
            predictionActions
        )
        // If activity prediction consistent, apply to page
        if (currentActivity == prevActivity) {
            if (currentActivity == "sitting") {
                runOnUiThread {
                    this@ViewActivity.predictedActivity.text =
                        String.format("Activity: sitting/standing")
                }
            } else {
                runOnUiThread {
                    this@ViewActivity.predictedActivity.text =
                        String.format("Activity: $currentActivity")
                }
            }
        } else {
            prevActivity = currentActivity
        }

        if (dynamicActivities.contains(currentActivity)) {
            runOnUiThread {
                this@ViewActivity.predictedBreathing.text = String.format("normal")
                setDrawable(context, currentActivity, "normal")
                if (currentActivity == "sitting" ) {
                    saveToDatabase("sitting/standing", "normal")
                }else{
                    saveToDatabase(currentActivity, "normal")
                }
            }
        } else {
            val currentBreathing: String = predict(
                secondModel,
                data,
                breathingAction
            )
            // if breathing prediction consistent , apply to page
            if (currentBreathing == prevBreathing) {
                runOnUiThread {
                    this@ViewActivity.predictedBreathing.text =
                        String.format("Breathing: $currentBreathing")
                    setDrawable(context, currentActivity, currentBreathing)
                    if (currentActivity == "sitting" ) {
                        saveToDatabase("sitting/standing", currentBreathing)
                    }else{
                        saveToDatabase(currentActivity, currentBreathing)
                    }
                }
            } else {
                prevBreathing = currentBreathing
            }
        }
    }

    /**
     * Set image corresponding to predicted activity and breathing
     * @param context
     * @param currentActivity predicted activity
     * @param currentBreathing predicted breathing
     */
    private fun setDrawable(context: Context, currentActivity: String, currentBreathing: String) {
        try {
            val imageName =
                String.format(currentActivity.replace(" ", "_") + "_" + currentBreathing)
            val drawable = AppCompatResources.getDrawable(
                context,
                context.resources.getIdentifier(imageName, "drawable", context.packageName)
            )
            visualRep.setImageDrawable(drawable)
        } catch (e :Exception){
            Log.d("Image not found", "$currentActivity + $currentBreathing")
        }

    }

    /**
     * Register respeck reciever
     * @param handlerRespeck respeck handler
     */
    private fun registerReciever(handlerRespeck: Handler) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            this.registerReceiver(
                respeckLiveUpdateReceiver,
                filterTestRespeck,
                null,
                handlerRespeck,
                RECEIVER_NOT_EXPORTED
            )

        } else {
            this.registerReceiver(
                respeckLiveUpdateReceiver,
                filterTestRespeck,
                null,
                handlerRespeck
            )
        }
    }

    /**
     * Give prediction based on model given, data and actions (labels of prediction)
     * @param model to use for prediction
     * @param dataPoints input for model
     * @param actions potential labels for model output
     */
    private fun predict(
        model: Interpreter,
        dataPoints: FloatArray,
        actions: ArrayList<String>
    ): String {
        // Set up reshape of data for ML model
        val inputFeatures: TensorBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 50, 6), DataType.FLOAT32)
        inputFeatures.loadArray(dataPoints)
        // Set up output array to store model's output
        val outputShape = model.getOutputTensor(0).shape()
        val output = Array(1) { FloatArray(outputShape[1]) }
        model.run(inputFeatures.buffer, output)
        val maxIndex = output[0].withIndex().maxByOrNull { outputPred -> outputPred.value }?.index
        return actions[maxIndex!!]
    }

    /** 
     * Load model to be used for prediction
     * @param model path of model's tflite file
     */
    private fun loadModel(model: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = this.assets.openFd(model)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset: Long = fileDescriptor.startOffset
        val declareLength: Long = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declareLength)
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            if (job != null) {
                job!!.cancel()
            }
            respeckDataBuffer.clear()
            unregisterReceiver(respeckLiveUpdateReceiver)
            handlerThreadRespeck.quitSafely()
            looperRespeck.quitSafely()
        } catch (e: Exception) {
            Log.d(
                "RESPECK RECIEVER",
                "Respeck reciever is not registered, Error message: ${e.message}"
            )
        }
        try {
//                    firestoreDb.terminate()
            firstModel.close()
            secondModel.close()
        } catch (e: Exception) {
            Log.d("MODELS", "Model are already closed, Error message: ${e.message}")
        }
    }

    private fun saveToDatabase(currentActivity: String, currentBreathing: String ){
        val userId = FirebaseAuth.getInstance().currentUser?.uid
        if (userId == null) {
            Log.e("Firestore", "User not logged in")
            return
        }

        // Check if the activity or breathing has changed
        if (currentActivity != lastSavedActivity || currentBreathing != lastSavedBreathing) {
            val data = hashMapOf(
                "timestamp" to System.currentTimeMillis(),
                "activity" to currentActivity,
                "breathing" to currentBreathing
            )

            firestoreDb.collection("users")
                .document(userId)
                .collection("activityData")
                .add(data)
                .addOnSuccessListener {
                    Log.d("Firestore", "Data saved successfully")
                    // Update the last saved data
                    lastSavedActivity = currentActivity
                    lastSavedBreathing = currentBreathing
                }
                .addOnFailureListener { e ->
                    Log.e("Firestore", "Error saving data", e)
                }
        }
    }

}

/**
 * Add data from Respeck to stream of data
 * @param liveData from Respeck
 */
private fun MutableList<Float>.addLiveData(liveData: RESpeckLiveData) {
    // Make space for new data coming from Respeck
    this.add(liveData.accelX)
    this.add(liveData.accelY)
    this.add(liveData.accelZ)
    this.add(liveData.gyro.x)
    this.add(liveData.gyro.y)
    this.add(liveData.gyro.z)
}


