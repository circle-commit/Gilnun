// Runs the app's pure-Kotlin code (guidance, detector, text reading) on the JVM, without
// the Android SDK. Use run_tests.sh, which also generates the guidance fixtures.
plugins {
    kotlin("jvm") version "2.0.21"
}

val appSources = "../app/src/main/java/com/kojingapla/glass"

sourceSets {
    main {
        kotlin {
            setSrcDirs(listOf(appSources))
            include(
                "ApproachTracker.kt",
                "DuplicateTextSuppressor.kt",
                "GuidanceEngine.kt",
                "Models.kt",
                "ObjectDetector.kt",
                "SceneAnalyzer.kt",
                "SceneDetection.kt",
                "TextReading.kt"
            )
        }
    }
}

dependencies {
    // The JVM build of the runtime the app uses (onnxruntime-android); same Java API.
    implementation("com.microsoft.onnxruntime:onnxruntime:1.22.0")
    testImplementation(kotlin("test"))
    testImplementation("org.json:json:20240303")
}

tasks.test {
    useJUnitPlatform()
    maxHeapSize = "2g"
    for (name in listOf("gilnun.fixtures", "gilnun.model", "gilnun.golden", "gilnun.repoRoot")) {
        System.getProperty(name)?.let { systemProperty(name, it) }
    }
    testLogging {
        events("failed")
        showStandardStreams = true
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
