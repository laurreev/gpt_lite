@echo off
echo GPT Lite - Build Verification
echo ==============================

echo Step 1: Checking Flutter installation...
flutter --version
if errorlevel 1 (
    echo ERROR: Flutter not found. Please install Flutter SDK.
    exit /b 1
)

echo.
echo Step 2: Checking project dependencies...
flutter pub get
if errorlevel 1 (
    echo ERROR: Failed to get dependencies.
    exit /b 1
)

echo.
echo Step 3: Analyzing code...
flutter analyze
if errorlevel 1 (
    echo WARNING: Code analysis found issues. Check the output above.
)

echo.
echo Step 4: Building debug APK...
flutter build apk --debug
if errorlevel 1 (
    echo ERROR: Failed to build debug APK.
    exit /b 1
)

echo.
echo ✅ Build verification completed successfully!
echo.
echo The debug APK is available at:
echo build\app\outputs\flutter-apk\app-debug.apk
echo.
echo Next steps:
echo 1. Install the APK on your Android device
echo 2. Test the demo mode functionality
echo 3. Follow the integration guide to add real llama.cpp support
echo.
pause
