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

If NDK 27.0.12077973 is not available, these are good alternatives:

1. **NDK 26.1.10909125** (Previous stable version)
2. **NDK 25.1.8937393** (Older but stable)
3. **NDK 23.2.8568313** (Widely compatible)

To use a different version, update `build.gradle.kts`:

```kotlin
android {
    ndkVersion = "26.1.10909125"  // or your preferred version
}
```

## Troubleshooting

### Common Issues:

1. **NDK not found**: Make sure ANDROID_HOME is set correctly
2. **Build errors**: Try cleaning the project: `flutter clean`
3. **Version conflicts**: Use the exact version specified in build.gradle.kts
4. **Space issues**: NDK downloads are ~1.5GB, ensure sufficient disk space

### Checking Current NDK Installation:

```bash
flutter doctor -v
```

This will show your current Android toolchain and NDK status.

## Why NDK 27.0.12077973?

- ✅ **Latest Features**: Most recent NDK with latest optimizations
- ✅ **Plugin Compatibility**: Required by current Flutter plugins
- ✅ **Enhanced C++20 Support**: Modern C++ standard support
- ✅ **Better Performance**: Latest compiler optimizations
- ✅ **Backward Compatible**: Works with older native code
- ✅ **llama.cpp Ready**: Excellent support for modern C++ features

## Plugin Compatibility

This NDK version is required by the following Flutter plugins in your project:
- `file_picker`
- `flutter_plugin_android_lifecycle`
- `path_provider_android`
- `permission_handler_android`
- `shared_preferences_android`

Using a consistent NDK version across all plugins prevents build conflicts.
