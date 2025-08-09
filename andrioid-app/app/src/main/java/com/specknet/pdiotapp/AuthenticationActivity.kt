package com.specknet.pdiotapp

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentManager
import androidx.fragment.app.FragmentPagerAdapter
import androidx.viewpager.widget.ViewPager
import com.google.firebase.auth.FirebaseAuth
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.android.gms.auth.api.signin.GoogleSignInOptions
import com.google.android.gms.common.api.ApiException
import com.google.firebase.auth.GoogleAuthProvider

class AuthenticationActivity : AppCompatActivity() {

    private lateinit var auth: FirebaseAuth
    private lateinit var emailEditText: EditText
    private lateinit var passwordEditText: EditText
    private lateinit var loginButton: Button
    private lateinit var signUpTextView: TextView // Changed from EditText to TextView
    private lateinit var googleSignIn: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_authentication)

        auth = FirebaseAuth.getInstance()


        // Initialize UI components by linking to their XML ids
        emailEditText = findViewById(R.id.editTextEmail)
        passwordEditText = findViewById(R.id.editTextPassword)
        loginButton = findViewById(R.id.buttonLogin)
        signUpTextView = findViewById(R.id.textViewSignUp) // Corrected the ID and type to TextView
        googleSignIn = findViewById(R.id.buttonGoogleSignIn)

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
                loginButton.visibility = View.VISIBLE
                signUpTextView.visibility = View.VISIBLE
                googleSignIn.visibility = View.VISIBLE
            }
        }

        // Add AuthStateListener when the Activity starts
        auth.addAuthStateListener(authStateListener)

        // Handle the login button click
        loginButton.setOnClickListener {
            val email = emailEditText.text.toString().trim()
            val password = passwordEditText.text.toString().trim()
            signInWithEmailAndPassword(email, password)
        }

        // Handle the sign-up text click
        signUpTextView.setOnClickListener {
            // Intent to navigate to the sign-up activity
            val signUpIntent = Intent(this, SignUpActivity::class.java)
            startActivity(signUpIntent)
        }


        // Configure Google Sign-In
        val gso = GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
            .requestIdToken(getString(R.string.default_web_client_id))
            .requestEmail()
            .build()

        val googleSignInClient = GoogleSignIn.getClient(this, gso)

        googleSignIn.setOnClickListener {
            val signInIntent = googleSignInClient.signInIntent
            startActivityForResult(signInIntent, RC_SIGN_IN)
        }

    }


    private fun signInWithEmailAndPassword(email: String, password: String) {
        if (email.isNotEmpty() && password.isNotEmpty()) {
            auth.signInWithEmailAndPassword(email, password).addOnCompleteListener(this) { task ->
                if (task.isSuccessful) {
                    // Sign-in success
                    navigateToMainActivity()
                } else {
                    // If sign in fails, display a message to the user
                    Toast.makeText(this, "Authentication failed: ${task.exception?.message}", Toast.LENGTH_SHORT).show()
                }
            }
        } else {
            Toast.makeText(this, "Please enter email and password.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        // Result returned from launching the Intent from GoogleSignInApi.getSignInIntent(...);
        if (requestCode == RC_SIGN_IN) {
            val task = GoogleSignIn.getSignedInAccountFromIntent(data)
            try {
                // Google Sign-In was successful, authenticate with Firebase
                val account = task.getResult(ApiException::class.java)!!
                firebaseAuthWithGoogle(account.idToken!!)
            } catch (e: ApiException) {
                // Google Sign-In failed
                Log.w("AuthenticationActivity", "Google sign in failed", e)
                Toast.makeText(this, "Google sign in failed: " + e.localizedMessage, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun firebaseAuthWithGoogle(idToken: String) {
        val credential = GoogleAuthProvider.getCredential(idToken, null)
        auth.signInWithCredential(credential)
            .addOnCompleteListener(this) { task ->
                if (task.isSuccessful) {
                    // Sign in success, update UI with the signed-in user's information
                    val user = auth.currentUser
                    navigateToMainActivity()
                } else {
                    // If sign in fails, display a message to the user.
                    Log.w("AuthenticationActivity", "signInWithCredential:failure", task.exception)
                }
            }
    }

    private fun navigateToMainActivity() {
        val mainActivityIntent = Intent(this, MainActivity::class.java)
        startActivity(mainActivityIntent)
        finish()
    }


    companion object {
        private const val RC_SIGN_IN = 9001 // Google sign-in request code
        private const val TAG = "AuthenticationActivity"
    }
}
