# Flutter ONNX Runtime GenAI Plugin - CMake Configuration
# 
# This CMakeLists.txt builds the Flutter FFI bridge and links against
# the ONNX Runtime GenAI shared library.
#
# The Flutter tooling requires CMake 3.10 or later.
# Do not increase this version to maintain compatibility.

cmake_minimum_required(VERSION 3.10)

project(flutter_onnxruntime_genai_library VERSION 0.1.0 LANGUAGES CXX)

# =============================================================================
# Build Options
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# Source Files
# =============================================================================

add_library(flutter_onnxruntime_genai SHARED
  "flutter_onnxruntime_genai.cpp"
)

# =============================================================================
# Include Directories
# =============================================================================

target_include_directories(flutter_onnxruntime_genai PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# If using the submodule headers directly
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../native_src/onnxruntime-genai/src")
  target_include_directories(flutter_onnxruntime_genai PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../native_src/onnxruntime-genai/src
  )
endif()

# =============================================================================
# Library Properties
# =============================================================================

set_target_properties(flutter_onnxruntime_genai PROPERTIES
  PUBLIC_HEADER flutter_onnxruntime_genai.h
  OUTPUT_NAME "flutter_onnxruntime_genai"
  # Ensure symbols are exported
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

target_compile_definitions(flutter_onnxruntime_genai PUBLIC DART_SHARED_LIB)

# =============================================================================
# Platform-specific Configuration
# =============================================================================

if(ANDROID)
  # -------------------------------------------------------------------------
  # Android Configuration
  # -------------------------------------------------------------------------
  
  # CRITICAL: Android 15 requires 16KB page alignment
  # This flag ensures the shared library is compatible with Android 15+
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
  target_link_options(flutter_onnxruntime_genai PRIVATE "-Wl,-z,max-page-size=16384")
  
  # Define path to prebuilt ONNX Runtime GenAI libraries
  set(ONNXRUNTIME_GENAI_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../android/src/main/jniLibs/${ANDROID_ABI}")
  
  # Link against ONNX Runtime GenAI
  if(EXISTS "${ONNXRUNTIME_GENAI_LIB_DIR}/libonnxruntime-genai.so")
    add_library(onnxruntime-genai SHARED IMPORTED)
    set_target_properties(onnxruntime-genai PROPERTIES
      IMPORTED_LOCATION "${ONNXRUNTIME_GENAI_LIB_DIR}/libonnxruntime-genai.so"
    )
    target_link_libraries(flutter_onnxruntime_genai PRIVATE onnxruntime-genai)
  else()
    message(STATUS "ONNX Runtime GenAI library not found at: ${ONNXRUNTIME_GENAI_LIB_DIR}")
    message(STATUS "Build will succeed but linking will fail at runtime without the library.")
    message(STATUS "Run ./scripts/build_onnx_libs.sh to build the native libraries.")
  endif()
  
  # Link Android-specific libraries
  target_link_libraries(flutter_onnxruntime_genai PRIVATE
    log
    android
  )

elseif(APPLE)
  # -------------------------------------------------------------------------
  # iOS/macOS Configuration (handled primarily by Podspec)
  # -------------------------------------------------------------------------
  
  # The iOS build is handled by CocoaPods/Xcode
  # This section is for reference and potential macOS desktop support
  
  set(ONNXRUNTIME_GENAI_FRAMEWORK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ios/Frameworks")
  
  if(EXISTS "${ONNXRUNTIME_GENAI_FRAMEWORK_DIR}/onnxruntime-genai.xcframework")
    message(STATUS "Found ONNX Runtime GenAI XCFramework")
    # XCFramework linking is handled by the Podspec on iOS
  endif()

else()
  # -------------------------------------------------------------------------
  # Linux/Desktop Configuration
  # -------------------------------------------------------------------------
  
  # For desktop development/testing
  find_library(ONNXRUNTIME_GENAI_LIB onnxruntime-genai
    HINTS
      /usr/local/lib
      /usr/lib
      ${CMAKE_CURRENT_SOURCE_DIR}/../native_src/onnxruntime-genai/build
  )
  
  if(ONNXRUNTIME_GENAI_LIB)
    target_link_libraries(flutter_onnxruntime_genai PRIVATE ${ONNXRUNTIME_GENAI_LIB})
  else()
    message(STATUS "ONNX Runtime GenAI library not found for desktop build")
  endif()
  
  # Link pthread for std::mutex
  find_package(Threads REQUIRED)
  target_link_libraries(flutter_onnxruntime_genai PRIVATE Threads::Threads)
  
endif()

# =============================================================================
# Compiler Warnings
# =============================================================================

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(flutter_onnxruntime_genai PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
  )
endif()
