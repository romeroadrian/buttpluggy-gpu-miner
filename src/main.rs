use buttpluggy_gpu_miner::Config;
use std::env;
use std::process;

fn main() {
    println!("Buttpluggy GPU miner");

    let config = Config::new(env::args()).unwrap_or_else(|err| {
        eprintln!("Failed parsing arguments: {err}");
        process::exit(1);
    });

    if let Err(e) = buttpluggy_gpu_miner::gpu(config) {
        eprintln!("GPU application error: {e}");
        process::exit(1);
    }
}
