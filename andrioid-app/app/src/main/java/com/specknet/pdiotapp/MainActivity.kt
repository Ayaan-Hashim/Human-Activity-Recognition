package com.specknet.pdiotapp

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.widget.Button
import androidx.appcompat.app.AlertDialog
import androidx.core.app.ActivityCompat
import com.google.android.material.snackbar.Snackbar
import com.google.firebase.auth.FirebaseAuth
import com.specknet.pdiotapp.bluetooth.BluetoothSpeckService
import com.specknet.pdiotapp.bluetooth.ConnectingActivity
import com.specknet.pdiotapp.live.LiveDataActivity
import com.specknet.pdiotapp.live.ViewActivity
import com.specknet.pdiotapp.onboarding.OnBoardingActivity
import com.specknet.pdiotapp.utils.Constants
import com.specknet.pdiotapp.utils.Utils
import com.specknet.pdiotapp.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    // buttons and textviews
    lateinit var liveProcessingButton: Button
    lateinit var viewActivityButton: Button
    lateinit var pairingButton: Button
    lateinit var viewHistory: Button

    private lateinit var binding: ActivityMainBinding

    // permissions
    lateinit var permissionAlertDialog: AlertDialog.Builder

    val permissionsForRequest = arrayListOf<String>()

    var locationPermissionGranted = false
    var cameraPermissionGranted = false
    var readStoragePermissionGranted = false
    var writeStoragePermissionGranted = false
    var BLUETOOTH  = false
    var BLUETOOTH_ADMIN = false
    var BLUETOOTH_SCAN = false
    var BLUETOOTH_CONNECT = false
    var BLUETOOTH_ADVERTISE = false

    // broadcast receiver
    val filter = IntentFilter()

    var isUserFirstTime = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val auth = FirebaseAuth.getInstance()
        if (auth.currentUser == null) {
            // Start AuthenticationActivity for result
            val loginIntent = Intent(this, AuthenticationActivity::class.java)
            startActivity(loginIntent)
        } else {
            // User is authenticated, initialize the main view
            initializeMainView()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == LOGIN_REQUEST_CODE) {
            if (resultCode == RESULT_OK) {
                // User authenticated, initialize the main view
                initializeMainView()
            } else {
                // Authentication failed, close the app or handle accordingly
                finish()
            }
        }
    }

    private fun initializeMainView() {
        setContentView(R.layout.activity_main)

        // check whether the onboarding screen should be shown
        val sharedPreferences = getSharedPreferences(Constants.PREFERENCES_FILE, Context.MODE_PRIVATE)
        if (sharedPreferences.contains(Constants.PREF_USER_FIRST_TIME)) {
            isUserFirstTime = false
        }
        else {
            isUserFirstTime = true
            sharedPreferences.edit().putBoolean(Constants.PREF_USER_FIRST_TIME, false).apply()
            val introIntent = Intent(this, OnBoardingActivity::class.java)
            startActivity(introIntent)
        }

        liveProcessingButton = findViewById(R.id.live_button)
        viewActivityButton = findViewById(R.id.view_activity)
        pairingButton = findViewById(R.id.ble_button)
        viewHistory = findViewById(R.id.view_history)

        permissionAlertDialog = AlertDialog.Builder(this)

        setupClickListeners()

        setupPermissions()

        setupBluetoothService()

        // register a broadcast receiver for respeck status
        filter.addAction(Constants.ACTION_RESPECK_CONNECTED)
        filter.addAction(Constants.ACTION_RESPECK_DISCONNECTED)

    }

    fun setupClickListeners() {
        liveProcessingButton.setOnClickListener {
            val intent = Intent(this, LiveDataActivity::class.java)
            startActivity(intent)
        }

        viewActivityButton.setOnClickListener{
            val intent = Intent(this, ViewActivity::class.java)
            startActivity(intent)
        }

        pairingButton.setOnClickListener {
            val intent = Intent(this, ConnectingActivity::class.java)
            startActivity(intent)
        }

        viewHistory.setOnClickListener {
            val intent = Intent(this, HistoryActivity::class.java)
            startActivity(intent)
        }
    }

    fun setupPermissions() {
        // request permissions

        // location permission
        Log.i("Permissions", "Location permission = " + locationPermissionGranted)
        if (ActivityCompat.checkSelfPermission(applicationContext,
                Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            permissionsForRequest.add(Manifest.permission.ACCESS_FINE_LOCATION)
            permissionsForRequest.add(Manifest.permission.ACCESS_COARSE_LOCATION)
        }
        else {
            locationPermissionGranted = true
        }

        // camera permission
        Log.i("Permissions", "Camera permission = " + cameraPermissionGranted)
        if (ActivityCompat.checkSelfPermission(applicationContext,
                Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Log.i("Permissions", "Camera permission = " + cameraPermissionGranted)
            permissionsForRequest.add(Manifest.permission.CAMERA)
        }
        else {
            cameraPermissionGranted = true
        }

        // read storage permission
        Log.i("Permissions", "Read st permission = " + readStoragePermissionGranted)
        if (ActivityCompat.checkSelfPermission(applicationContext,
                Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            Log.i("Permissions", "Read st permission = " + readStoragePermissionGranted)
            permissionsForRequest.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        else {
            readStoragePermissionGranted = true
        }

        // write storage permission
        Log.i("Permissions", "Write storage permission = " + writeStoragePermissionGranted)
        if (ActivityCompat.checkSelfPermission(applicationContext,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            Log.i("Permissions", "Write storage permission = " + writeStoragePermissionGranted)
            permissionsForRequest.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
        else {
            writeStoragePermissionGranted = true
        }






        if (ActivityCompat.checkSelfPermission(applicationContext,
                Manifest.permission.BLUETOOTH) != PackageManager.PERMISSION_GRANTED) {
            Log.i("Permissions", "Write storage permission = " + BLUETOOTH)
            permissionsForRequest.add(Manifest.permission.BLUETOOTH)
        }
        else {
            BLUETOOTH = true
        }

        if (ActivityCompat.checkSelfPermission(applicationContext,
                Manifest.permission.BLUETOOTH_ADMIN) != PackageManager.PERMISSION_GRANTED) {
            Log.i("Permissions", "Write storage permission = " + BLUETOOTH_ADMIN)
            permissionsForRequest.add(Manifest.permission.BLUETOOTH_ADMIN)
        }
        else {
            BLUETOOTH_ADMIN = true
        }

        if (ActivityCompat.checkSelfPermission(applicationContext,
                Manifest.permission.BLUETOOTH_SCAN) != PackageManager.PERMISSION_GRANTED) {
            Log.i("Permissions", "Write storage permission = " + BLUETOOTH_SCAN)
            permissionsForRequest.add(Manifest.permission.BLUETOOTH_SCAN)
        }
        else {
            BLUETOOTH_SCAN = true
        }

        if (ActivityCompat.checkSelfPermission(applicationContext,
                Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) {
            Log.i("Permissions", "Write storage permission = " + BLUETOOTH_CONNECT)
            permissionsForRequest.add(Manifest.permission.BLUETOOTH_CONNECT)
        }
        else {
            BLUETOOTH_CONNECT = true
        }

        if (ActivityCompat.checkSelfPermission(applicationContext,
                Manifest.permission.BLUETOOTH_ADVERTISE) != PackageManager.PERMISSION_GRANTED) {
            Log.i("Permissions", "Write storage permission = " + BLUETOOTH_ADVERTISE)
            permissionsForRequest.add(Manifest.permission.BLUETOOTH_ADVERTISE)
        }
        else {
            BLUETOOTH_ADVERTISE = true
        }

        if (permissionsForRequest.size >= 1) {
            ActivityCompat.requestPermissions(this,
                permissionsForRequest.toTypedArray(),
                Constants.REQUEST_CODE_PERMISSIONS)
        }

    }

    fun setupBluetoothService() {
        val isServiceRunning = Utils.isServiceRunning(BluetoothSpeckService::class.java, applicationContext)
        Log.i("debug","isServiceRunning = " + isServiceRunning)

        // check sharedPreferences for an existing Respeck id
        val sharedPreferences = getSharedPreferences(Constants.PREFERENCES_FILE, Context.MODE_PRIVATE)
        if (sharedPreferences.contains(Constants.RESPECK_MAC_ADDRESS_PREF)) {
            Log.i("sharedpref", "Already saw a respeckID, starting service and attempting to reconnect")

            // launch service to reconnect
            // start the bluetooth service if it's not already running
            if(!isServiceRunning) {
                Log.i("service", "Starting BLT service")
                val simpleIntent = Intent(this, BluetoothSpeckService::class.java)
                this.startService(simpleIntent)
            }
        }
        else {
            Log.i("sharedpref", "No Respeck seen before, must pair first")
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(requestCode == Constants.REQUEST_CODE_PERMISSIONS) {
            if(grantResults.isNotEmpty()) {
                for (i in grantResults.indices) {
                    when(permissionsForRequest[i]) {
                        Manifest.permission.ACCESS_COARSE_LOCATION -> locationPermissionGranted = true
                        Manifest.permission.ACCESS_FINE_LOCATION -> locationPermissionGranted = true
                        Manifest.permission.CAMERA -> cameraPermissionGranted = true
                        Manifest.permission.READ_EXTERNAL_STORAGE -> readStoragePermissionGranted = true
                        Manifest.permission.WRITE_EXTERNAL_STORAGE -> writeStoragePermissionGranted = true
                        Manifest.permission.BLUETOOTH -> BLUETOOTH = true
                        Manifest.permission.BLUETOOTH_ADMIN -> BLUETOOTH_ADMIN = true
                        Manifest.permission.BLUETOOTH_SCAN -> BLUETOOTH_SCAN = true
                        Manifest.permission.BLUETOOTH_CONNECT -> BLUETOOTH_CONNECT = true
                        Manifest.permission.BLUETOOTH_ADVERTISE -> BLUETOOTH_ADVERTISE = true
                    }

                }
            }
        }

        // count how many permissions need granting
        var numberOfPermissionsUngranted = 0
        if (!locationPermissionGranted) numberOfPermissionsUngranted++
        if (!cameraPermissionGranted) numberOfPermissionsUngranted++
        if (!readStoragePermissionGranted) numberOfPermissionsUngranted++
        if (!writeStoragePermissionGranted) numberOfPermissionsUngranted++

        // show a general message if we need multiple permissions
        if (numberOfPermissionsUngranted > 1) {
            val generalSnackbar = Snackbar
                .make(binding.coordinatorLayout, "Several permissions are needed for correct app functioning", Snackbar.LENGTH_LONG)
                .setAction("SETTINGS") {
                    startActivity(Intent(Settings.ACTION_SETTINGS))
                }
                .show()
        }
        else if(numberOfPermissionsUngranted == 1) {
            var snackbar: Snackbar = Snackbar.make(binding.coordinatorLayout, "", Snackbar.LENGTH_LONG)
            if (!locationPermissionGranted) {
                snackbar = Snackbar
                    .make(
                        binding.coordinatorLayout,
                        "Location permission needed for Bluetooth to work.",
                        Snackbar.LENGTH_LONG
                    )
            }

            if(!cameraPermissionGranted) {
                snackbar = Snackbar
                    .make(
                        binding.coordinatorLayout,
                        "Camera permission needed for QR code scanning to work.",
                        Snackbar.LENGTH_LONG
                    )
            }

            if(!readStoragePermissionGranted || !writeStoragePermissionGranted) {
                snackbar = Snackbar
                    .make(
                        binding.coordinatorLayout,
                        "Storage permission needed to record sensor.",
                        Snackbar.LENGTH_LONG
                    )
            }

            snackbar.setAction("SETTINGS") {
                val settingsIntent = Intent(Settings.ACTION_SETTINGS)
                startActivity(settingsIntent)
            }
                .show()
        }

    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        val id = item.itemId
        if (id == R.id.show_tutorial) {
            val introIntent = Intent(this, OnBoardingActivity::class.java)
            startActivity(introIntent)
            return true
        }
        if (id == R.id.sign_out) {
            signOut()
        }

        return super.onOptionsItemSelected(item)
    }

    private fun signOut() {
        FirebaseAuth.getInstance().signOut()
        val signOutIntent = Intent(this, AuthenticationActivity::class.java)
        signOutIntent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(signOutIntent)
        finish()
    }

    companion object {
        const val LOGIN_REQUEST_CODE = 1 // This is the request code you will use for startActivityForResult
    }

}