use buttpluggy_gpu_miner::{Config, gpu};
use buttpluggy_gpu_miner::web3_utils::{ContractClient, query_salt, query_current_difficulty};
use std::env;
use std::process;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Buttpluggy GPU miner");

    let client = ContractClient::new(
        "https://eth.llamarpc.com",
        "0x0000420538CD5AbfBC7Db219B6A1d125f5892Ab0",
    )?;

    let salt_bytes = query_salt(&client).await?;
    let salt: [u8; 32] = salt_bytes
        .to_vec()
        .try_into()
        .map_err(|_| "invalid length for salt from contract")?;

    let difficulty_u256 = query_current_difficulty(&client).await?;
    let difficulty: u8 = difficulty_u256
        .try_into()
        .map_err(|_| "difficulty value from contract is too large for u8")?;

    //let config = Config::new(std::env::args());
    let config = Config::new(env::args(), salt, difficulty).unwrap_or_else(|err| {
        eprintln!("Failed parsing arguments: {err}");
        process::exit(1);
    });

    if let Err(e) = gpu(&client, config).await.map_err(|e| Box::<dyn std::error::Error>::from(e)) {
        eprintln!("GPU application error: {e}");
        process::exit(1);
    }
    
    Ok(())
}