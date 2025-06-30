use ethers::prelude::*;
use std::sync::Arc;

abigen!(
    HuffPlug,
    r#"[
        {
            "constant": true,
            "inputs": [
                {
                    "name": "tokenId",
                    "type": "uint256"
                }
            ],
            "name": "ownerOf",
            "outputs": [
                {
                    "name": "",
                    "type": "address"
                }
            ],
            "payable": false,
            "stateMutability": "view",
            "type": "function"
        },
        {
            "constant": true,
            "inputs": [],
            "name": "currentDifficulty",
            "outputs": [
                {
                    "name": "",
                    "type": "uint256"
                }
            ],
            "payable": false,
            "stateMutability": "view",
            "type": "function"
        },
        {
            "constant": true,
            "inputs": [],
            "name": "salt",
            "outputs": [
                {
                    "name": "",
                    "type": "bytes32"
                }
            ],
            "payable": false,
            "stateMutability": "view",
            "type": "function"
        }
    ]"#
);

// Estructura para almacenar el cliente y el contrato
pub struct ContractClient {
    contract: HuffPlug<Provider<Http>>,
}

impl ContractClient {
    pub fn new(rpc_url: &str, contract_address: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let provider = Provider::<Http>::try_from(rpc_url)?;
        let client = Arc::new(provider);
        let address: Address = contract_address.parse()?;
        let contract = HuffPlug::new(address, client.clone());
        Ok(ContractClient { contract })
    }
}

// Función para consultar ownerOf
pub async fn have_owner(
    client: &ContractClient,
    token_id: U256,
) -> Result<bool, Box<dyn std::error::Error>> {
    let owner = client.contract.owner_of(token_id).call().await?;
    Ok(owner != Address::zero())
}

// Función para consultar currentDifficulty, devuelve el valor uint256
pub async fn query_current_difficulty(
    client: &ContractClient,
) -> Result<U256, Box<dyn std::error::Error>> {
    let difficulty = client.contract.current_difficulty().call().await?;
    Ok(difficulty)
}

// Función para consultar salt, devuelve el valor bytes32 como Bytes
pub async fn query_salt(
    client: &ContractClient,
) -> Result<Bytes, Box<dyn std::error::Error>> {
    let salt = client.contract.salt().call().await?;
    Ok(Bytes::from(salt.to_vec()))
}