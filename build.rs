fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Set ORT_STRATEGY to system by default to avoid prebuilt triplet issues
    unsafe {
        std::env::set_var("ORT_STRATEGY", "system");
    }

    // Get the path to our custom ONNX Runtime SDK
    let sdk_path = std::env::var("ORT_LIB_LOCATION")
        .unwrap_or_else(|_| {
            // Fallback to default location
            format!("{}/../lib/onnxruntime_sdk", std::env::var("CARGO_MANIFEST_DIR").unwrap())
        });

    println!("cargo:warning=Using ONNX Runtime SDK from: {}", sdk_path);

    // Set include path
    let include_path = format!("{}/include", sdk_path);
    println!("cargo:include={}", include_path);

    // Set library search path
    let lib_path = format!("{}/lib", sdk_path);
    println!("cargo:rustc-link-search=native={}", lib_path);

    // Link against onnxruntime
    println!("cargo:rustc-link-lib=dylib=onnxruntime");

    // Emit cargo metadata to help with debugging
    println!("cargo:warning=Include directory: {}", include_path);
    println!("cargo:warning=Lib directory: {}", lib_path);
}