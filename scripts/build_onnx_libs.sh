#!/bin/bash
#
# build_onnx_libs.sh
# 
# Cross-compilation script for ONNX Runtime GenAI native libraries.
# Builds for Android (arm64-v8a, x86_64) and iOS (device + simulator).
#
# Prerequisites:
#   - Android NDK installed and ANDROID_NDK_HOME or ANDROID_NDK set
#   - Xcode command line tools installed
#   - CMake 3.20+ installed
#   - Python 3.8+ installed (for ONNX Runtime build scripts)
#
# Usage:
#   ./scripts/build_onnx_libs.sh [android|ios|all]
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SUBMODULE_DIR="$PROJECT_ROOT/native_src/onnxruntime-genai"
BUILD_DIR="$PROJECT_ROOT/build"

# Android configuration
ANDROID_MIN_SDK=24
ANDROID_ARCHS=("arm64-v8a" "x86_64")
ANDROID_OUTPUT_DIR="$PROJECT_ROOT/android/src/main/jniLibs"

# iOS configuration
IOS_MIN_VERSION="13.0"
IOS_OUTPUT_DIR="$PROJECT_ROOT/ios/Frameworks"

# Build configuration
CMAKE_BUILD_TYPE="Release"
PARALLEL_JOBS=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if submodule exists
    if [ ! -d "$SUBMODULE_DIR/.git" ] && [ ! -f "$SUBMODULE_DIR/CMakeLists.txt" ]; then
        log_error "ONNX Runtime GenAI submodule not found at: $SUBMODULE_DIR"
        log_info "Please run: git submodule update --init --recursive"
        exit 1
    fi
    
    # Check CMake
    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found. Please install CMake 3.20+"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    log_info "Prerequisites check passed."
}

check_android_ndk() {
    # Find Android NDK
    if [ -n "$ANDROID_NDK_HOME" ]; then
        NDK_PATH="$ANDROID_NDK_HOME"
    elif [ -n "$ANDROID_NDK" ]; then
        NDK_PATH="$ANDROID_NDK"
    elif [ -d "$HOME/Library/Android/sdk/ndk" ]; then
        # Find latest NDK version
        NDK_PATH=$(find "$HOME/Library/Android/sdk/ndk" -maxdepth 1 -type d | sort -V | tail -1)
    else
        log_error "Android NDK not found. Set ANDROID_NDK_HOME or ANDROID_NDK environment variable."
        exit 1
    fi
    
    if [ ! -f "$NDK_PATH/build/cmake/android.toolchain.cmake" ]; then
        log_error "Invalid Android NDK path: $NDK_PATH"
        exit 1
    fi
    
    log_info "Using Android NDK: $NDK_PATH"
}

# =============================================================================
# Android Build
# =============================================================================

build_android() {
    log_info "=========================================="
    log_info "Building ONNX Runtime GenAI for Android"
    log_info "=========================================="
    
    check_android_ndk
    
    for ARCH in "${ANDROID_ARCHS[@]}"; do
        log_info "Building for Android $ARCH..."
        
        ANDROID_BUILD_DIR="$BUILD_DIR/android-$ARCH"
        mkdir -p "$ANDROID_BUILD_DIR"
        
        # Map architecture to Android ABI
        case $ARCH in
            "arm64-v8a")
                ANDROID_ABI="arm64-v8a"
                ;;
            "x86_64")
                ANDROID_ABI="x86_64"
                ;;
            *)
                log_warn "Unknown architecture: $ARCH, skipping..."
                continue
                ;;
        esac
        
        cd "$ANDROID_BUILD_DIR"
        
        # Configure with CMake
        # CRITICAL: Include 16KB page alignment flag for Android 15+
        cmake "$SUBMODULE_DIR" \
            -DCMAKE_TOOLCHAIN_FILE="$NDK_PATH/build/cmake/android.toolchain.cmake" \
            -DANDROID_ABI="$ANDROID_ABI" \
            -DANDROID_PLATFORM="android-$ANDROID_MIN_SDK" \
            -DANDROID_NDK="$NDK_PATH" \
            -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,max-page-size=16384" \
            -DCMAKE_EXE_LINKER_FLAGS="-Wl,-z,max-page-size=16384" \
            -DORT_GENAI_BUILD_TESTS=OFF \
            -DORT_GENAI_BUILD_EXAMPLES=OFF \
            -DUSE_CUDA=OFF \
            -DUSE_ROCM=OFF \
            -DENABLE_PYTHON=OFF \
            -DENABLE_MODEL_BENCHMARK=OFF \
            -DBUILD_SHARED_LIBS=ON \
            || {
                log_error "CMake configuration failed for $ARCH"
                exit 1
            }
        
        # Build
        cmake --build . --config "$CMAKE_BUILD_TYPE" -j "$PARALLEL_JOBS" || {
            log_error "Build failed for $ARCH"
            exit 1
        }
        
        # Copy output to jniLibs
        OUTPUT_LIB_DIR="$ANDROID_OUTPUT_DIR/$ARCH"
        mkdir -p "$OUTPUT_LIB_DIR"
        
        # Find and copy the built shared library
        if [ -f "libonnxruntime-genai.so" ]; then
            cp "libonnxruntime-genai.so" "$OUTPUT_LIB_DIR/"
            log_info "Copied libonnxruntime-genai.so to $OUTPUT_LIB_DIR/"
        elif [ -f "lib/libonnxruntime-genai.so" ]; then
            cp "lib/libonnxruntime-genai.so" "$OUTPUT_LIB_DIR/"
            log_info "Copied libonnxruntime-genai.so to $OUTPUT_LIB_DIR/"
        else
            log_warn "Could not find libonnxruntime-genai.so in build output"
            find . -name "*.so" -type f
        fi
        
        cd "$PROJECT_ROOT"
    done
    
    log_info "Android build complete!"
}

# =============================================================================
# iOS Build
# =============================================================================

build_ios() {
    log_info "=========================================="
    log_info "Building ONNX Runtime GenAI for iOS"
    log_info "=========================================="
    
    # Check for Xcode
    if ! command -v xcodebuild &> /dev/null; then
        log_error "Xcode command line tools not found. Please install Xcode."
        exit 1
    fi
    
    # Create build directories
    IOS_DEVICE_BUILD_DIR="$BUILD_DIR/ios-device"
    IOS_SIM_BUILD_DIR="$BUILD_DIR/ios-simulator"
    XCFRAMEWORK_BUILD_DIR="$BUILD_DIR/xcframework"
    
    mkdir -p "$IOS_DEVICE_BUILD_DIR" "$IOS_SIM_BUILD_DIR" "$XCFRAMEWORK_BUILD_DIR"
    
    # -------------------------------------------------------------------------
    # Build for iOS Device (arm64)
    # -------------------------------------------------------------------------
    log_info "Building for iOS Device (arm64)..."
    
    cd "$IOS_DEVICE_BUILD_DIR"
    
    cmake "$SUBMODULE_DIR" \
        -G "Xcode" \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_ARCHITECTURES="arm64" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN_VERSION" \
        -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DORT_GENAI_BUILD_TESTS=OFF \
        -DORT_GENAI_BUILD_EXAMPLES=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_ROCM=OFF \
        -DENABLE_PYTHON=OFF \
        -DENABLE_MODEL_BENCHMARK=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        || {
            log_error "CMake configuration failed for iOS device"
            exit 1
        }
    
    cmake --build . --config "$CMAKE_BUILD_TYPE" -- -sdk iphoneos || {
        log_error "Build failed for iOS device"
        exit 1
    }
    
    # -------------------------------------------------------------------------
    # Build for iOS Simulator (arm64 + x86_64)
    # -------------------------------------------------------------------------
    log_info "Building for iOS Simulator (arm64, x86_64)..."
    
    cd "$IOS_SIM_BUILD_DIR"
    
    cmake "$SUBMODULE_DIR" \
        -G "Xcode" \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN_VERSION" \
        -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DORT_GENAI_BUILD_TESTS=OFF \
        -DORT_GENAI_BUILD_EXAMPLES=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_ROCM=OFF \
        -DENABLE_PYTHON=OFF \
        -DENABLE_MODEL_BENCHMARK=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        || {
            log_error "CMake configuration failed for iOS simulator"
            exit 1
        }
    
    cmake --build . --config "$CMAKE_BUILD_TYPE" -- -sdk iphonesimulator || {
        log_error "Build failed for iOS simulator"
        exit 1
    }
    
    # -------------------------------------------------------------------------
    # Create XCFramework
    # -------------------------------------------------------------------------
    log_info "Creating XCFramework..."
    
    # Find the built frameworks/libraries
    DEVICE_LIB=$(find "$IOS_DEVICE_BUILD_DIR" -name "*.framework" -o -name "libonnxruntime-genai.a" | head -1)
    SIM_LIB=$(find "$IOS_SIM_BUILD_DIR" -name "*.framework" -o -name "libonnxruntime-genai.a" | head -1)
    
    if [ -z "$DEVICE_LIB" ] || [ -z "$SIM_LIB" ]; then
        log_warn "Could not find built libraries, attempting alternative approach..."
        
        # Try to find in Release subdirectory
        DEVICE_LIB=$(find "$IOS_DEVICE_BUILD_DIR/Release-iphoneos" -name "*.a" 2>/dev/null | head -1)
        SIM_LIB=$(find "$IOS_SIM_BUILD_DIR/Release-iphonesimulator" -name "*.a" 2>/dev/null | head -1)
    fi
    
    if [ -n "$DEVICE_LIB" ] && [ -n "$SIM_LIB" ]; then
        mkdir -p "$IOS_OUTPUT_DIR"
        
        # Check if it's a framework or static library
        if [[ "$DEVICE_LIB" == *.framework ]]; then
            xcodebuild -create-xcframework \
                -framework "$DEVICE_LIB" \
                -framework "$SIM_LIB" \
                -output "$IOS_OUTPUT_DIR/onnxruntime-genai.xcframework"
        else
            # For static libraries, we need to create a framework structure first
            log_info "Creating framework from static libraries..."
            
            DEVICE_FRAMEWORK_DIR="$XCFRAMEWORK_BUILD_DIR/device/onnxruntime-genai.framework"
            SIM_FRAMEWORK_DIR="$XCFRAMEWORK_BUILD_DIR/simulator/onnxruntime-genai.framework"
            
            mkdir -p "$DEVICE_FRAMEWORK_DIR/Headers"
            mkdir -p "$SIM_FRAMEWORK_DIR/Headers"
            
            # Copy static library
            cp "$DEVICE_LIB" "$DEVICE_FRAMEWORK_DIR/onnxruntime-genai"
            cp "$SIM_LIB" "$SIM_FRAMEWORK_DIR/onnxruntime-genai"
            
            # Copy headers
            if [ -d "$SUBMODULE_DIR/src" ]; then
                cp "$SUBMODULE_DIR/src/ort_genai_c.h" "$DEVICE_FRAMEWORK_DIR/Headers/" 2>/dev/null || true
                cp "$SUBMODULE_DIR/src/ort_genai_c.h" "$SIM_FRAMEWORK_DIR/Headers/" 2>/dev/null || true
            fi
            
            # Create Info.plist
            cat > "$DEVICE_FRAMEWORK_DIR/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>onnxruntime-genai</string>
    <key>CFBundleIdentifier</key>
    <string>com.microsoft.onnxruntime-genai</string>
    <key>CFBundleName</key>
    <string>onnxruntime-genai</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>MinimumOSVersion</key>
    <string>13.0</string>
</dict>
</plist>
PLIST
            cp "$DEVICE_FRAMEWORK_DIR/Info.plist" "$SIM_FRAMEWORK_DIR/Info.plist"
            
            # Create XCFramework
            xcodebuild -create-xcframework \
                -framework "$DEVICE_FRAMEWORK_DIR" \
                -framework "$SIM_FRAMEWORK_DIR" \
                -output "$IOS_OUTPUT_DIR/onnxruntime-genai.xcframework"
        fi
        
        log_info "XCFramework created at: $IOS_OUTPUT_DIR/onnxruntime-genai.xcframework"
    else
        log_error "Could not find built libraries for XCFramework creation"
        log_info "Device lib search path: $IOS_DEVICE_BUILD_DIR"
        log_info "Simulator lib search path: $IOS_SIM_BUILD_DIR"
        find "$BUILD_DIR" -name "*.a" -o -name "*.framework" 2>/dev/null
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    log_info "iOS build complete!"
}

# =============================================================================
# Cleanup
# =============================================================================

clean_build() {
    log_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    log_info "Build directory cleaned."
}

# =============================================================================
# Main
# =============================================================================

print_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  android    Build for Android only"
    echo "  ios        Build for iOS only"
    echo "  all        Build for both platforms (default)"
    echo "  clean      Remove build artifacts"
    echo "  help       Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  ANDROID_NDK_HOME   Path to Android NDK"
    echo "  ANDROID_NDK        Alternative path to Android NDK"
    echo ""
    echo "Examples:"
    echo "  $0 android    # Build Android libraries only"
    echo "  $0 ios        # Build iOS XCFramework only"
    echo "  $0            # Build for all platforms"
}

main() {
    COMMAND="${1:-all}"
    
    case $COMMAND in
        android)
            check_prerequisites
            build_android
            ;;
        ios)
            check_prerequisites
            build_ios
            ;;
        all)
            check_prerequisites
            build_android
            build_ios
            ;;
        clean)
            clean_build
            ;;
        help|--help|-h)
            print_usage
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
