# NDK Installation Guide

## Installing Android NDK 27.0.12077973

### Method 1: Android Studio (Recommended)

1. **Open Android Studio**
2. **Go to SDK Manager**:
   - File → Settings → Appearance & Behavior → System Settings → Android SDK
   - Or click the SDK Manager icon in the toolbar
3. **Navigate to SDK Tools tab**
4. **Find NDK (Side by side)**:
   - Check the checkbox next to "NDK (Side by side)"
   - Click "Show Package Details"
   - Select version "27.0.12077973"
5. **Click Apply** and let it download

### Method 2: Command Line

```bash
# Using Android SDK Manager
sdkmanager "ndk;27.0.12077973"

# On Windows (if sdkmanager is in PATH)
sdkmanager.bat "ndk;27.0.12077973"
```

### Method 3: Manual Download

1. Go to [Android NDK Downloads](https://developer.android.com/ndk/downloads)
2. Download NDK r27b (27.0.12077973)
3. Extract to your Android SDK directory under `ndk/27.0.12077973/`

## Verification

### Check if NDK is installed:

**Windows:**
```cmd
dir "%ANDROID_HOME%\ndk\27.0.12077973"
```

**Linux/Mac:**
```bash
ls $ANDROID_HOME/ndk/27.0.12077973
```

### Set Environment Variables (if needed):

**Windows:**
```cmd
set ANDROID_NDK_HOME=%ANDROID_HOME%\ndk\27.0.12077973
set NDK_HOME=%ANDROID_HOME%\ndk\27.0.12077973
```

**Linux/Mac:**
```bash
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/27.0.12077973
export NDK_HOME=$ANDROID_HOME/ndk/27.0.12077973
```

## Alternative NDK Versions

If NDK 25.1.8937393 is not available, these are good alternatives:

1. **NDK 23.2.8568313** (Stable, widely used)
2. **NDK 24.0.8215888** (Good C++17 support)
3. **NDK 26.1.10909125** (Latest, if you want cutting-edge features)

To use a different version, update `build.gradle.kts`:

```kotlin
android {
    ndkVersion = "23.2.8568313"  // or your preferred version
}
```

## Troubleshooting

### Common Issues:

1. **NDK not found**: Make sure ANDROID_HOME is set correctly
2. **Build errors**: Try cleaning the project: `flutter clean`
3. **Version conflicts**: Use the exact version specified in build.gradle.kts
4. **Space issues**: NDK downloads are ~1GB, ensure sufficient disk space

### Checking Current NDK Installation:

```bash
flutter doctor -v
```

This will show your current Android toolchain and NDK status.

## Why NDK 25.1.8937393?

- ✅ **Stable and mature**
- ✅ **Great C++17 support** (needed for llama.cpp)
- ✅ **Compatible with Flutter**
- ✅ **Well-tested with native libraries**
- ✅ **Good performance optimizations**
