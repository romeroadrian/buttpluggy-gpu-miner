#![warn(unused_crate_dependencies, unreachable_pub)]
#![deny(unused_must_use, rust_2018_idioms)]

use alloy_primitives::{hex, FixedBytes};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use console::Term;
use fs4::FileExt;
use ocl::{Buffer, Context, Device, MemFlags, Platform, ProQue, Program, Queue};
use rand::{thread_rng, Rng};
use separator::Separatable;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};
use tiny_keccak::{Hasher, Keccak};

// workset size (tweak this!)
const WORK_SIZE: u32 = 0x4000000; // max. 0x15400000 to abs. max 0xffffffff

const WORK_FACTOR: u128 = (WORK_SIZE as u128) / 1_000_000;

static KERNEL_SRC: &str = include_str!("./kernels/keccak256.cl");

pub struct Config {
    pub calling_address: [u8; 20],
    pub salt: [u8; 32],
    pub difficulty: u8,
    pub gpu_device: u8,
}

impl Config {
    pub fn new(mut args: std::env::Args) -> Result<Self, &'static str> {
        // get args, skipping first arg (program name)
        args.next();

        let Some(calling_address_string) = args.next() else {
            return Err("didn't get a calling_address argument");
        };
        let Some(salt_string) = args.next() else {
            return Err("didn't get a salt argument");
        };
        let Some(difficulty_string) = args.next() else {
            return Err("didn't get a difficulty argument");
        };
        let gpu_device_string = match args.next() {
            Some(arg) => arg,
            None => String::from("0"), // default to device 0
        };

        // convert main arguments from hex string to vector of bytes
        let Ok(calling_address_vec) = hex::decode(calling_address_string) else {
            return Err("could not decode calling address argument");
        };
        let Ok(salt_vec) = hex::decode(salt_string) else {
            return Err("could not decode salt argument");
        };

        // convert from vector to fixed array
        let Ok(calling_address) = calling_address_vec.try_into() else {
            return Err("invalid length for calling address argument");
        };
        let Ok(salt) = salt_vec.try_into() else {
            return Err("invalid length for salt argument");
        };

        // convert gpu arguments to u8 values
        let Ok(gpu_device) = gpu_device_string.parse::<u8>() else {
            return Err("invalid gpu device value");
        };
        let Ok(difficulty) = difficulty_string.parse::<u8>() else {
            return Err("invalid difficulty value supplied");
        };

        if difficulty > 20 {
            return Err("invalid value for difficulty argument. (valid: 0..=20)");
        }

        Ok(Self {
            calling_address,
            salt,
            gpu_device,
            difficulty,
        })
    }
}

pub fn gpu(config: Config) -> ocl::Result<()> {
    // (create if necessary) and open a file where found salts will be written
    let file = output_file();

    // track how many addresses have been found and information about them
    let mut found: u64 = 0;
    let mut found_list: Vec<String> = vec![];

    // set up a controller for terminal output
    let term = Term::stdout();

    // set up a platform to use
    let platform = Platform::new(ocl::core::default_platform()?);

    // set up the device to use
    let device = Device::by_idx_wrap(platform, config.gpu_device as usize)?;

    // set up the context to use
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    // set up the program to use
    let program = Program::builder()
        .devices(device)
        .src(mk_kernel_src(&config))
        .build(&context)?;

    // set up the queue to use
    let queue = Queue::new(&context, device, None)?;

    // set up the "proqueue" (or amalgamation of various elements) to use
    let ocl_pq = ProQue::new(context, queue, program, Some(WORK_SIZE));

    // create a random number generator
    let mut rng = thread_rng();

    // determine the start time
    let start_time: f64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // set up variables for tracking performance
    let mut rate: f64 = 0.0;
    let mut cumulative_nonce: u64 = 0;

    // the previous timestamp of printing to the terminal
    let mut previous_time: f64 = 0.0;

    // the last work duration in milliseconds
    let mut work_duration_millis: u64 = 0;

    // begin searching for addresses
    loop {
        // construct the 4-byte message to hash, leaving last 8 of salt empty
        let salt = FixedBytes::<4>::random();

        // build a corresponding buffer for passing the message to the kernel
        let message_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(4)
            .copy_host_slice(&salt[..])
            .build()?;

        // reset nonce & create a buffer to view it in little-endian
        // for more uniformly distributed nonces, we shall initialize it to a random value
        let mut nonce: [u32; 1] = rng.gen();
        let mut view_buf = [0; 8];

        // build a corresponding buffer for passing the nonce to the kernel
        let mut nonce_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(1)
            .copy_host_slice(&nonce)
            .build()?;

        // establish a buffer for nonces that result in desired addresses
        let mut solutions: Vec<u64> = vec![0; 1];
        let solutions_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().write_only())
            .len(1)
            .copy_host_slice(&solutions)
            .build()?;

        // repeatedly enqueue kernel to search for new addresses
        loop {
            // build the kernel and define the type of each buffer
            let kern = ocl_pq
                .kernel_builder("hashMessage")
                .arg_named("message", None::<&Buffer<u8>>)
                .arg_named("nonce", None::<&Buffer<u32>>)
                .arg_named("solutions", None::<&Buffer<u64>>)
                .build()?;

            // set each buffer
            kern.set_arg("message", Some(&message_buffer))?;
            kern.set_arg("nonce", Some(&nonce_buffer))?;
            kern.set_arg("solutions", &solutions_buffer)?;

            // enqueue the kernel
            unsafe { kern.enq()? };

            // calculate the current time
            let mut now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let current_time = now.as_secs() as f64;

            // we don't want to print too fast
            let print_output = current_time - previous_time > 0.99;
            previous_time = current_time;

            // clear the terminal screen
            if print_output {
                term.clear_screen()?;

                // get the total runtime and parse into hours : minutes : seconds
                let total_runtime = current_time - start_time;
                let total_runtime_hrs = total_runtime as u64 / 3600;
                let total_runtime_mins = (total_runtime as u64 - total_runtime_hrs * 3600) / 60;
                let total_runtime_secs = total_runtime
                    - (total_runtime_hrs * 3600) as f64
                    - (total_runtime_mins * 60) as f64;

                // determine the number of attempts being made per second
                let work_rate: u128 = WORK_FACTOR * cumulative_nonce as u128;
                if total_runtime > 0.0 {
                    rate = 1.0 / total_runtime;
                }

                // fill the buffer for viewing the properly-formatted nonce
                LittleEndian::write_u64(&mut view_buf, (nonce[0] as u64) << 32);

                // calculate the terminal height, defaulting to a height of ten rows
                let height = terminal_size().map(|(_w, Height(h))| h).unwrap_or(10);

                // display information about the total runtime and work size
                term.write_line(&format!(
                    "total runtime: {}:{:02}:{:02} ({} cycles)\t\t\t\
                   work size per cycle: {}",
                    total_runtime_hrs,
                    total_runtime_mins,
                    total_runtime_secs,
                    cumulative_nonce,
                    WORK_SIZE.separated_string(),
                ))?;

                // display information about the attempt rate and found solutions
                term.write_line(&format!(
                    "rate: {:.2} million attempts per second\t\t\t\
                   total found this run: {}",
                    work_rate as f64 * rate,
                    found
                ))?;

                // display information about the current search criteria
                term.write_line(&format!(
                    "current search space: {}xxxxxxxx{:08x}\t\t\
                   difficulty: {}",
                    hex::encode(salt),
                    BigEndian::read_u64(&view_buf),
                    config.difficulty,
                ))?;

                // display recently found solutions based on terminal height
                let rows = if height < 5 { 1 } else { height as usize - 4 };
                let last_rows: Vec<String> = found_list.iter().cloned().rev().take(rows).collect();
                let ordered: Vec<String> = last_rows.iter().cloned().rev().collect();
                let recently_found = &ordered.join("\n");
                term.write_line(recently_found)?;
            }

            // increment the cumulative nonce (does not reset after a match)
            cumulative_nonce += 1;

            // record the start time of the work
            let work_start_time_millis = now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000;

            // sleep for 98% of the previous work duration to conserve CPU
            if work_duration_millis != 0 {
                std::thread::sleep(std::time::Duration::from_millis(
                    work_duration_millis * 980 / 1000,
                ));
            }

            // read the solutions from the device
            solutions_buffer.read(&mut solutions).enq()?;

            // record the end time of the work and compute how long the work took
            now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            work_duration_millis = (now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000)
                - work_start_time_millis;

            // if at least one solution is found, end the loop
            if solutions[0] != 0 {
                break;
            }

            // if no solution has yet been found, increment the nonce
            nonce[0] += 1;

            // update the nonce buffer with the incremented nonce value
            nonce_buffer = Buffer::builder()
                .queue(ocl_pq.queue().clone())
                .flags(MemFlags::new().read_write())
                .len(1)
                .copy_host_slice(&nonce)
                .build()?;
        }

        // iterate over each solution, first converting to a fixed array
        for &solution in &solutions {
            if solution == 0 {
                continue;
            }

            let solution = solution.to_le_bytes();

            let mut buttpluggy_nonce = [0; 32];
            // 0-20 are zeroes
            // 20-24 is the local salt
            buttpluggy_nonce[20..24].copy_from_slice(&salt[..]);
            // 24-32 is the nonce from the gpu
            buttpluggy_nonce[24..].copy_from_slice(&solution);

            let mut solution_message = [0; 84];
            // 0-20 caller address
            solution_message[0..20].copy_from_slice(&config.calling_address);
            // 20-52 buttpluggy salt
            solution_message[20..52].copy_from_slice(&config.salt);
            // 52-84 is the buttpluggy nonce
            solution_message[52..].copy_from_slice(&buttpluggy_nonce);

            // create new hash object
            let mut hash = Keccak::v256();

            // update with header
            hash.update(&solution_message);

            // hash the payload and get the result
            let mut res: [u8; 32] = [0; 32];
            hash.finalize(&mut res);

            let output = format!(
                "0x{} -> 0x{}",
                hex::encode(buttpluggy_nonce),
                hex::encode(res),
            );

            found_list.push(output.to_string());

            file.lock_exclusive().expect("Couldn't lock file.");

            writeln!(&file, "{output}").expect("Couldn't write to `buttpluggy_nonces.txt` file.");

            file.unlock().expect("Couldn't unlock file.");
            found += 1;
        }
    }
}

#[track_caller]
fn output_file() -> File {
    OpenOptions::new()
        .append(true)
        .create(true)
        .read(true)
        .open("buttpluggy_nonces.txt")
        .expect("Could not create or open `buttpluggy_nonces.txt` file.")
}

/// Creates the OpenCL kernel source code by populating the template with the
/// values from the Config object.
fn mk_kernel_src(config: &Config) -> String {
    let mut src = String::with_capacity(2048 + KERNEL_SRC.len());

    let caller = config.calling_address.iter();
    let salt = config.salt.iter();
    for (i, x) in caller.chain(salt).enumerate() {
        writeln!(src, "#define S_{} {}u", i, x).unwrap();
    }
    let difficulty = config.difficulty;
    writeln!(src, "#define LEADING_ZEROES {difficulty}").unwrap();

    src.push_str(KERNEL_SRC);

    src
}
