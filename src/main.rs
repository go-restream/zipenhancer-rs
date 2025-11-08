//! ZipEnhancer Rust Version Main Program
//!
//! This is the command line entry point for ZipEnhancer audio denoise processor.
//! Provides user-friendly interface to configure and run audio denoise processing.

use clap::Parser;
use std::process;
use std::path::Path;
use zipenhancer::{init_logging, Args, Result};

fn main() {
    let args = Args::parse();

    if let Err(e) = check_and_setup_onnx_library(&args) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }

    init_logging(args.verbose);

    if let Err(e) = run(args) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    if args.verbose {
        let info = zipenhancer::get_library_info();
        println!("{}", info);
        println!();
    }

    if args.test_only {
        return run_test_mode();
    }

    let config = zipenhancer::config::Config::from_args_and_config(args)?;

    if !config.input_path.exists() {
        return Err(zipenhancer::ZipEnhancerError::config(format!(
            "Input file does not exist: {}",
            config.input_path.display()
        )));
    }

    if !config.model_path().exists() {
        return Err(zipenhancer::ZipEnhancerError::config(format!(
            "Model file does not exist: {}",
            config.model_path().display()
        )));
    }

    if let Some(parent) = config.output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!("=== ZipEnhancer Rust Version Audio Denoise Processor ===");
    println!("Input file: {}", config.input_path.display());
    println!("Output file: {}", config.output_path.display());
    println!("Model file: {}", config.model_path().display());
    println!("Sample rate: {} Hz", config.sample_rate());
    println!("Overlap ratio: {:.1}%", config.overlap_ratio() * 100.0);
    println!("Segment size: {} samples", config.segment_size());
    println!("Verbose mode: {}", if config.verbose() { "Enabled" } else { "Disabled" });
    println!("Performance monitoring: {}", if config.enable_performance_monitoring() { "Enabled" } else { "Disabled" });
    println!("===========================================");
    println!();

    let mut processor = zipenhancer::processing::AudioProcessor::new(config.clone())?;
    processor.warm_up()?;

    if config.verbose() {
        processor.print_status();
        println!();
    }

    let result = processor.process_file(&config.input_path, &config.output_path)?;

    println!("=== Processing Complete ===");
    println!("Input file: {}", result.input_path.display());
    println!("Output file: {}", result.output_path.display());
    println!("Processing time: {:.2} seconds", result.performance_metrics.processing_time_seconds);
    println!("Real-time factor (RTF): {:.3}", result.performance_metrics.real_time_factor);

    if config.verbose() {
        println!("Input audio duration: {:.2} seconds", result.performance_metrics.input_duration_seconds);
        println!("Processed segments: {}", result.performance_metrics.segment_count);
        if result.performance_metrics.average_inference_time_ms > 0.0 {
            println!("Average inference time: {:.2} ms", result.performance_metrics.average_inference_time_ms);
        }
        println!("===================");
    }

    Ok(())
}

fn run_test_mode() -> Result<()> {
    println!("=== Test Mode ===");
    println!("Validating basic functionality...");

    let config = zipenhancer::config::Config::default();
    println!("✅ Default config creation successful");

    config.validate()?;
    println!("✅ Config validation passed");

    let cpu_count = zipenhancer::config::utils::cpu_count();
    println!("✅ Detected {} CPU cores", cpu_count);

    let info = zipenhancer::get_library_info();
    println!("✅ Library info: {} v{}", info.name, info.version);

    println!();
    println!("All basic tests passed! Project structure is ready.");
    println!("Next step: implement audio I/O and ONNX inference functions.");

    Ok(())
}

/// Check and setup ONNX Runtime library path
fn check_and_setup_onnx_library(args: &Args) -> Result<()> {
    // If user specified library path, check if file exists
    if let Some(ref lib_path) = args.onnx_lib {
        if !lib_path.exists() {
            return Err(zipenhancer::ZipEnhancerError::config(format!(
                "Specified ONNX Runtime library file does not exist: {}\nPlease check if path is correct, or use --onnx-lib parameter to specify correct library file path",
                lib_path.display()
            )));
        }

        // Set library path environment variable
        unsafe {
            std::env::set_var("LD_LIBRARY_PATH",
                format!("{}:{}",
                    lib_path.parent().unwrap_or_else(|| Path::new(".")).display(),
                    std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
                ));
        }

        if args.verbose {
            println!("Using specified ONNX Runtime library: {}", lib_path.display());
        }
        return Ok(());
    }

    // Try to find common ONNX Runtime library locations
    let common_paths = vec![
        "/opt/homebrew/lib/libonnxruntime.dylib",
        "/usr/local/lib/libonnxruntime.dylib",
        "/usr/lib/libonnxruntime.dylib",
        "/lib/libonnxruntime.dylib",
        "lib/libonnxruntime.1.24.0.dylib",
        "lib/libonnxruntime.dylib",
        "lib/onnxruntime_sdk/lib/libonnxruntime.dylib",
    ];

    let mut found_path = None;
    for path in &common_paths {
        if Path::new(path).exists() {
            found_path = Some(Path::new(path));
            break;
        }
    }

    if let Some(found) = found_path {
        // Set library path environment variable
        unsafe {
            std::env::set_var("LD_LIBRARY_PATH",
                format!("{}:{}",
                    found.parent().unwrap_or_else(|| Path::new(".")).display(),
                    std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
                ));
        }

        if args.verbose {
            println!("Found ONNX Runtime library: {}", found.display());
        }
    } else {
        // Display friendly error message and help
        println!("=== ONNX Runtime Library Not Found ===");
        println!("Cannot find ONNX Runtime library file. Please ensure ONNX Runtime is installed or use --onnx-lib parameter to specify library file path.");
        println!();
        println!("Solutions:");
        println!("1. Install with Homebrew: brew install onnxruntime");
        println!("2. Download from official site: https://github.com/microsoft/onnxruntime/releases");
        println!("3. Use --onnx-lib parameter to specify library file path:");
        println!("   ./zipenhancer --onnx-lib /path/to/libonnxruntime.dylib [other parameters...]");
        println!();
        println!("Common library file locations:");
        for path in &common_paths {
            println!("  - {}", path);
        }
        println!("============================");
        println!();

        return Err(zipenhancer::ZipEnhancerError::config(
            "ONNX Runtime library not found. Please install ONNX Runtime or use --onnx-lib parameter to specify library file path."
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_test_mode() {
        let _args = Args {
            test_only: true,
            verbose: true,
            model: std::path::PathBuf::from("./model/ZipEnhancer_ONNX/ZipEnhancer.onnx"),
            input: std::path::PathBuf::from("input.wav"),
            output: std::path::PathBuf::from("output.wav"),
            sample_rate: 16000,
            overlap: 0.1,
            segment_size: 16000,
            max_retries: 3,
            disable_performance_monitoring: false,
            inference_threads: 1,
            config_file: None,
            onnx_lib: None,
        };

        assert!(run_test_mode().is_ok());
    }
}