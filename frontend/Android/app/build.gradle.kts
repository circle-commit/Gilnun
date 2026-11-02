plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.kojingapla.glass"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.kojingapla.glass"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    androidResources {
        // The detector model is read whole at startup; stored uncompressed it loads faster.
        noCompress += "onnx"
    }
}

dependencies {
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.camera:camera-core:1.4.1")
    implementation("androidx.camera:camera-camera2:1.4.1")
    implementation("androidx.camera:camera-lifecycle:1.4.1")
    implementation("androidx.camera:camera-view:1.4.1")
    implementation("com.google.mlkit:text-recognition-korean:16.0.1")
    // On-device object detection. 1.31 adds usage telemetry and the INTERNET permission,
    // which this app does without; 1.22 has neither.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
}
