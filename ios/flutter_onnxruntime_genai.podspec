#
# Flutter ONNX Runtime GenAI Plugin - CocoaPods Specification
#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint flutter_onnxruntime_genai.podspec` to validate before publishing.
#

Pod::Spec.new do |s|
  s.name             = 'flutter_onnxruntime_genai'
  s.version          = '0.1.0'
  s.summary          = 'Flutter FFI plugin for ONNX Runtime GenAI multimodal inference.'
  s.description      = <<-DESC
A Flutter FFI plugin that wraps the Microsoft ONNX Runtime GenAI C-API
for on-device multimodal inference, specifically designed for models like
Phi-3.5 Vision on iOS devices.
                       DESC
  s.homepage         = 'https://github.com/colymba/flutter_onnxruntime_genai'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Thierry François' => 'thierry@colymba.fr' }

  # Source configuration
  s.source           = { :path => '.' }
  
  # =============================================================================
  # Source Files
  # =============================================================================
  
  # Include the C++ bridge source files
  # The forwarder in Classes/ includes files from ../src/
  s.source_files     = 'Classes/**/*', '../src/**/*.{h,cpp}'
  
  # Public headers that should be exposed
  s.public_header_files = '../src/flutter_onnxruntime_genai.h'
  
  # =============================================================================
  # Dependencies
  # =============================================================================
  
  s.dependency 'Flutter'
  
  # =============================================================================
  # Platform Configuration
  # =============================================================================
  
  s.platform = :ios, '13.0'
  
  # =============================================================================
  # Vendored Frameworks
  # =============================================================================
  
  # Include the ONNX Runtime GenAI XCFramework
  # The XCFramework should contain both device (arm64) and simulator slices
  s.vendored_frameworks = 'Frameworks/onnxruntime-genai.xcframework'
  
  # =============================================================================
  # Build Settings
  # =============================================================================
  
  s.pod_target_xcconfig = {
    # Module definition
    'DEFINES_MODULE' => 'YES',
    
    # Exclude i386 architecture for simulator (Flutter.framework doesn't contain it)
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    
    # C++ language settings
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',
    
    # Header search paths
    'HEADER_SEARCH_PATHS' => [
      '$(PODS_TARGET_SRCROOT)/../src',
      '$(PODS_TARGET_SRCROOT)/../src/include',
      '$(PODS_TARGET_SRCROOT)/../native_src/onnxruntime-genai/src'
    ].join(' '),
    
    # Framework search paths for the vendored XCFramework
    'FRAMEWORK_SEARCH_PATHS' => '$(PODS_TARGET_SRCROOT)/Frameworks',
    
    # Other linker flags
    'OTHER_LDFLAGS' => '-ObjC',
    
    # Enable full bitcode for release builds (optional, may be deprecated)
    # 'ENABLE_BITCODE' => 'NO',
  }
  
  # User target settings (applied to the app consuming this pod)
  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }
  
  # =============================================================================
  # Resource Bundles (if any)
  # =============================================================================
  
  # Uncomment if you have resources to bundle
  # s.resource_bundles = {
  #   'flutter_onnxruntime_genai' => ['Assets/**/*']
  # }
  
  # =============================================================================
  # Swift Configuration
  # =============================================================================
  
  s.swift_version = '5.0'
  
  # =============================================================================
  # Static Framework (optional)
  # =============================================================================
  
  # If you need this to be a static framework:
  # s.static_framework = true
  
end
