package com.specknet.pdiotapp

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.firebase.auth.FirebaseAuth

class SignUpActivity : AppCompatActivity() {

    private lateinit var auth: FirebaseAuth
    private lateinit var emailEditText: EditText
    private lateinit var passwordEditText: EditText
    private lateinit var confirmPasswordEditText: EditText
    private lateinit var signUpButton: Button
    private lateinit var linkLoginTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_sign_up)

        // Initialize Firebase Auth
        auth = FirebaseAuth.getInstance()

        // Auth state listener to handle user session
        val authStateListener = FirebaseAuth.AuthStateListener { firebaseAuth ->
            val user = firebaseAuth.currentUser
            if (user != null) {
                // User is signed in, navigate to the MainActivity
                navigateToMainActivity()
            } else {
                // User is signed out, show the login UI components
                emailEditText.visibility = View.VISIBLE
                passwordEditText.visibility = View.VISIBLE
                confirmPasswordEditText.visibility = View.VISIBLE
                signUpButton.visibility = View.VISIBLE
                linkLoginTextView.visibility = View.VISIBLE
            }
        }

        // Add AuthStateListener when the Activity starts
        auth.addAuthStateListener(authStateListener)

        // Link UI elements
        emailEditText = findViewById(R.id.editTextEmail)
        passwordEditText = findViewById(R.id.editTextPassword)
        confirmPasswordEditText = findViewById(R.id.editTextConfirmPassword)
        signUpButton = findViewById(R.id.buttonSignUp)
        linkLoginTextView = findViewById(R.id.textViewLinkLogin)

        // Set up listeners
        signUpButton.setOnClickListener { performSignUp() }
        linkLoginTextView.setOnClickListener { finish() } // Navigate back to the login screen

    }



    private fun performSignUp() {
        val email = emailEditText.text.toString().trim()
        val password = passwordEditText.text.toString()
        val confirmPassword = confirmPasswordEditText.text.toString()

        // Basic validation
        if (!validateForm(email, password, confirmPassword)) return

        // Firebase sign up
        auth.createUserWithEmailAndPassword(email, password)
            .addOnCompleteListener(this) { task ->
                if (task.isSuccessful) {
                    val user = auth.currentUser
                    Toast.makeText(baseContext, "Sign up successful.", Toast.LENGTH_SHORT).show()
                    // Sign-in success
                    navigateToMainActivity()
                } else {
                    Toast.makeText(baseContext, "Sign up failed: ${task.exception?.message}", Toast.LENGTH_SHORT).show()
                }
            }
    }

    private fun validateForm(email: String, password: String, confirmPassword: String): Boolean {
        if (email.isEmpty() || password.isEmpty() || confirmPassword.isEmpty()) {
            Toast.makeText(this, "Please fill in all fields.", Toast.LENGTH_SHORT).show()
            return false
        }

        if (password != confirmPassword) {
            Toast.makeText(this, "Passwords do not match.", Toast.LENGTH_SHORT).show()
            return false
        }

        // Additional validation can be added here (e.g., valid email format, password strength)

        return true
    }

    private fun navigateToMainActivity() {
        val mainActivityIntent = Intent(this, MainActivity::class.java)
        startActivity(mainActivityIntent)
        finish()
    }
}