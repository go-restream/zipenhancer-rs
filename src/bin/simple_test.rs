//! Simplified Test Program
//!
//! Used to test basic audio processing functions, does not depend on complex ONNX integration

use clap::Parser;
use std::process;
use zipenhancer::{init_logging, Result};

#[derive(Parser, Debug)]
#[command(name = "simple_test")]
#[command(about = "Simplified audio processing test program")]
struct Args {
    /// Input audio file
    #[arg(short, long)]
    input: String,

    /// Output audio file
    #[arg(short, long)]
    output: String,

    /// Sample rate
    #[arg(short, long, default_value = "16000")]
    sample_rate: u32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();
    init_logging(args.verbose);

    if let Err(e) = run(args) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    println!("=== Simplified Audio Processor Test ===");
    println!("Input file: {}", args.input);
    println!("Output file: {}", args.output);
    println!("Sample rate: {} Hz", args.sample_rate);
    println!("Verbose mode: {}", if args.verbose { "Enabled" } else { "Disabled" });
    println!("==========================");

    // Create simplified processor
    let mut processor = zipenhancer::simple_processor::SimpleAudioProcessor::new(
        args.sample_rate,
        args.verbose,
    );

    // Process audio file
    processor.process_file(
        std::path::Path::new(&args.input),
        std::path::Path::new(&args.output),
    )?;

    println!("Processing complete!");
    Ok(())
}