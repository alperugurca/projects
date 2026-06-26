plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.rigged.scienceoflosing"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.rigged.scienceoflosing"
        minSdk = 23
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
    }
}

kotlin {
    jvmToolchain(17)
}
