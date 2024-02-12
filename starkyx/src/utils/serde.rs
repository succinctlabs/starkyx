use plonky2::fri::proof::FriProofTarget;
use plonky2::hash::hash_types::{HashOutTarget, MerkleCapTarget};
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};
use serde::{Deserialize, Serialize};

pub trait BufferRead: Read {
    fn read_bytes(&mut self) -> IoResult<Vec<u8>> {
        let len = self.read_usize()?;
        let mut bytes = vec![0u8; len];
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

impl<'a> BufferRead for Buffer<'a> {}

pub trait BufferWrite: Write {
    fn write_bytes(&mut self, bytes: &[u8]) -> IoResult<()> {
        self.write_usize(bytes.len())?;
        self.write_all(bytes)
    }
}

impl BufferWrite for Vec<u8> {}

pub fn serialize_hash_out_target<S>(
    hash_out: &HashOutTarget,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    hash_out.elements.serialize(serializer)
}

pub fn deserialize_hash_out_target<'de, D>(deserializer: D) -> Result<HashOutTarget, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let elements = <[Target; 4]>::deserialize(deserializer)?;
    Ok(HashOutTarget { elements })
}

#[derive(Debug, Clone)]
pub struct SerdeHashOut(pub HashOutTarget);

impl Serialize for SerdeHashOut {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serialize_hash_out_target(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for SerdeHashOut {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserialize_hash_out_target(deserializer).map(Self)
    }
}

pub fn serialize_merkle_cap_target<S>(
    merkle_cap_target: &MerkleCapTarget,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let hashes = merkle_cap_target
        .0
        .iter()
        .map(|x| SerdeHashOut(*x))
        .collect::<Vec<_>>();
    hashes.serialize(serializer)
}

#[derive(Debug, Clone)]
pub struct SerdeMerkleCapTarget(pub MerkleCapTarget);

impl Serialize for SerdeMerkleCapTarget {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serialize_merkle_cap_target(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for SerdeMerkleCapTarget {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserialize_merkle_cap_target(deserializer).map(Self)
    }
}

pub fn deserialize_merkle_cap_target<'de, D>(deserializer: D) -> Result<MerkleCapTarget, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let hashes = Vec::<SerdeHashOut>::deserialize(deserializer)?;
    Ok(MerkleCapTarget(
        hashes.into_iter().map(|x| x.0).collect::<Vec<_>>(),
    ))
}

#[allow(clippy::ptr_arg)]
pub fn serialize_merkle_cap_targets<S>(
    merkle_cap_targets: &Vec<MerkleCapTarget>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    merkle_cap_targets
        .iter()
        .map(|x| SerdeMerkleCapTarget(x.clone()))
        .collect::<Vec<_>>()
        .serialize(serializer)
}

pub fn deserialize_merkle_cap_targets<'de, D>(
    deserializer: D,
) -> Result<Vec<MerkleCapTarget>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let hashes = Vec::<SerdeMerkleCapTarget>::deserialize(deserializer)?;
    Ok(hashes.into_iter().map(|x| x.0).collect::<Vec<_>>())
}

pub fn serialize_extension_target<S, const D: usize>(
    extension_target: &ExtensionTarget<D>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let elements = extension_target.0.to_vec();
    elements.serialize(serializer)
}

fn deserialize_extension_target<'de, D, const DEG: usize>(
    deserializer: D,
) -> Result<ExtensionTarget<DEG>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let elements = <Vec<Target>>::deserialize(deserializer)?;
    assert_eq!(elements.len(), DEG);
    Ok(ExtensionTarget(elements.try_into().unwrap()))
}

struct SerdeExtensionTarget<const D: usize>(pub ExtensionTarget<D>);

impl<const D: usize> Serialize for SerdeExtensionTarget<D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serialize_extension_target(&self.0, serializer)
    }
}

impl<'de, const DEG: usize> Deserialize<'de> for SerdeExtensionTarget<DEG> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserialize_extension_target(deserializer).map(Self)
    }
}

#[allow(clippy::ptr_arg)]
pub fn serialize_extension_targets<S, const D: usize>(
    extension_targets: &Vec<ExtensionTarget<D>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    extension_targets
        .iter()
        .map(|x| SerdeExtensionTarget(*x))
        .collect::<Vec<_>>()
        .serialize(serializer)
}

pub fn deserialize_extension_targets<'de, D, const DEG: usize>(
    deserializer: D,
) -> Result<Vec<ExtensionTarget<DEG>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let elements = Vec::<SerdeExtensionTarget<DEG>>::deserialize(deserializer)?;
    Ok(elements.into_iter().map(|x| x.0).collect::<Vec<_>>())
}

pub fn serialize_fri_proof_target<S, const D: usize>(
    fri_proof_target: &FriProofTarget<D>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mut buffer: Vec<u8> = Vec::new();
    buffer.write_target_fri_proof(fri_proof_target).unwrap();
    buffer.serialize(serializer)
}

pub fn deserialize_fri_proof_target<'de, D, const DEG: usize>(
    deserializer: D,
) -> Result<FriProofTarget<DEG>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let buffer = Vec::<u8>::deserialize(deserializer).unwrap();
    let mut buffer = Buffer::new(&buffer);
    Ok(buffer.read_target_fri_proof().unwrap())
}

pub fn serialize_fri_config<S>(
    fri_config: &plonky2::fri::FriConfig,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mut buffer: Vec<u8> = Vec::new();
    buffer.write_fri_config(fri_config).unwrap();
    buffer.serialize(serializer)
}

pub fn deserialize_fri_config<'de, D>(deserializer: D) -> Result<plonky2::fri::FriConfig, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let buffer = Vec::<u8>::deserialize(deserializer).unwrap();
    let mut buffer = Buffer::new(&buffer);
    Ok(buffer.read_fri_config().unwrap())
}
