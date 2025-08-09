package com.specknet.pdiotapp

import android.app.DatePickerDialog
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.Query
import java.text.SimpleDateFormat
import java.util.*

class HistoryActivity : AppCompatActivity() {

    private lateinit var tableLayout: TableLayout
    private lateinit var startDateButton: Button
    private lateinit var endDateButton: Button
    private lateinit var applyFilterButton: Button

    // Sample data
    private val activityData = listOf(
        "01-01-2023: Sitting",
        "02-02-2023: Shuffle Walking",
        "02-02-2023: Miscellaneous Movement",
        "02-02-2023: Lying Down Back",
        "03-03-2023: Lying Down Left"
    )

    private var startDate: Calendar? = null
    private var endDate: Calendar? = null

    private lateinit var firestoreDb: FirebaseFirestore

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)

        firestoreDb = FirebaseFirestore.getInstance()

        tableLayout = findViewById(R.id.tableLayoutActivity)
        startDateButton = findViewById(R.id.start_date)
        endDateButton = findViewById(R.id.end_date)
        applyFilterButton = findViewById(R.id.apply_filter)

        // Set listeners for date pickers
        startDateButton.setOnClickListener { showDatePickerDialog(isStartDate = true) }
        endDateButton.setOnClickListener { showDatePickerDialog(isStartDate = false) }

        applyFilterButton.setOnClickListener {
            if (startDate != null && endDate != null) {
                loadActivityDataFromFirestore()
            } else {
                Toast.makeText(this, "Please select start and end dates.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun showDatePickerDialog(isStartDate: Boolean) {
        val calendar = Calendar.getInstance()
        DatePickerDialog(
            this,
            R.style.CustomDatePickerDialog, // Make sure you have this style defined or remove this parameter
            { _, year, month, dayOfMonth ->
                val date = Calendar.getInstance().apply {
                    set(Calendar.YEAR, year)
                    set(Calendar.MONTH, month)
                    set(Calendar.DAY_OF_MONTH, dayOfMonth)
                }
                if (isStartDate) {
                    startDate = date
                    startDateButton.text = formatDate(date)
                } else {
                    endDate = date
                    endDateButton.text = formatDate(date)
                }
            },
            calendar.get(Calendar.YEAR),
            calendar.get(Calendar.MONTH),
            calendar.get(Calendar.DAY_OF_MONTH)
        ).show()
    }

    private fun formatDate(calendar: Calendar): String {
        val year = calendar.get(Calendar.YEAR).toString()
        val month = (calendar.get(Calendar.MONTH) + 1).toString().padStart(2, '0')
        val day = calendar.get(Calendar.DAY_OF_MONTH).toString().padStart(2, '0')
        return "$day-$month-$year"
    }

    private fun displayActivityData() {
        // Clear previous data
        tableLayout.removeAllViews()

        // Group activities by date
        val groupedData = activityData.groupBy { it.substringBefore(":") }

        // Sort dates to ensure the order
        val sortedDates = groupedData.keys.sorted()

        // Filter activity data based on selected date range
        val filteredDates = if (startDate != null && endDate != null) {
            val start = startDate!!.timeInMillis
            val end = endDate!!.timeInMillis
            sortedDates.filter { date ->
                val cal = Calendar.getInstance().apply {
                    val parts = date.split("-")
                    set(parts[2].toInt(), parts[1].toInt() - 1, parts[0].toInt())
                }
                val timeInMillis = cal.timeInMillis
                timeInMillis in start..end
            }
        } else {
            sortedDates
        }

        filteredDates.forEach { date ->
            val dateRow = TableRow(this).apply {
                layoutParams = TableLayout.LayoutParams(
                    TableLayout.LayoutParams.MATCH_PARENT,
                    TableLayout.LayoutParams.WRAP_CONTENT
                )
                setBackgroundColor(
                    ContextCompat.getColor(
                        this@HistoryActivity,
                        R.color.tableHeaderColor
                    )
                )
            }

            val dateView = TextView(this).apply {
                text = date
                setTypeface(null, Typeface.BOLD)
                gravity = Gravity.START
            }

            dateRow.addView(dateView)
            tableLayout.addView(dateRow)

            // Now display each activity for this date
            groupedData[date]?.forEachIndexed { index, record ->
                val activityRow = TableRow(this).apply {
                    layoutParams = TableLayout.LayoutParams(
                        TableLayout.LayoutParams.MATCH_PARENT,
                        TableLayout.LayoutParams.WRAP_CONTENT
                    )
                    background = if (index % 2 == 0) {
                        ContextCompat.getDrawable(this@HistoryActivity, R.color.tableEvenRowColor)
                    } else {
                        ContextCompat.getDrawable(this@HistoryActivity, R.color.tableOddRowColor)
                    }
                }

                val activityView = TextView(this).apply {
                    text = "    ${record.substringAfter(":").trim()}" // Indent activity text
                    gravity = Gravity.START
                    setPadding(50, 0, 0, 0) // Add padding to the left of the activity text
                }

                activityRow.addView(activityView)
                tableLayout.addView(activityRow)
            }
        }
    }
    private fun loadActivityDataFromFirestore() {
        val userId = FirebaseAuth.getInstance().currentUser?.uid
        if (userId != null) {
            val start = startDate?.timeInMillis ?: return
            val end = endDate?.timeInMillis ?: return

            firestoreDb.collection("users")
                .document(userId)
                .collection("activityData")
                .whereGreaterThanOrEqualTo("timestamp", start)
                .whereLessThanOrEqualTo("timestamp", end)
                .orderBy("timestamp", Query.Direction.DESCENDING)
                .get()
                .addOnSuccessListener { documents ->
                    for (document in documents) {
                        val timestamp = document.getLong("timestamp")
                        val activity = document.getString("activity") ?: "Unknown"
                        val breathing = document.getString("breathing") ?: "Unknown"
                        val date = SimpleDateFormat("dd-MM-yyyy", Locale.getDefault()).format(Date(timestamp ?: 0L))
                        addTableRow(date, activity, breathing)
                    }
                }
                .addOnFailureListener { exception ->
                    Toast.makeText(this, "Error getting documents: ${exception.message}", Toast.LENGTH_SHORT).show()
                }
        } else {
            Toast.makeText(this, "User not logged in", Toast.LENGTH_SHORT).show()
        }
    }


    private fun addTableRow(date: String, activity: String, breathing: String) {
        val tableRow = TableRow(this).apply {
            layoutParams = TableLayout.LayoutParams(
                TableLayout.LayoutParams.MATCH_PARENT,
                TableLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val dateTextView = TextView(this).apply {
            text = date
            // Apply any styling you want for the date cell
        }

        val activityTextView = TextView(this).apply {
            text = activity
            // Apply any styling you want for the activity cell
        }

        val breathingTextView = TextView(this).apply {
            text = breathing
            // Apply any styling you want for the breathing cell
        }

        // Add the TextViews to the TableRow
        tableRow.addView(dateTextView)
        tableRow.addView(activityTextView)
        tableRow.addView(breathingTextView)

        // Add the TableRow to the TableLayout
        tableLayout.addView(tableRow)
    }
}
