# StarkyX

<a title="Rama, CC BY-SA 3.0 FR &lt;https://creativecommons.org/licenses/by-sa/3.0/fr/deed.en&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Curta_type_I-CnAM_40092-IMG_6721-white.jpg"><img width="256" alt="Curta type I-CnAM 40092-IMG 6721-white" src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Curta_type_I-CnAM_40092-IMG_6721-white.jpg/256px-Curta_type_I-CnAM_40092-IMG_6721-white.jpg"></a>

## Introduction

StarkyX is a library for writing AIR constraints with composable and typed constraints. Currently, the main purpose of the library is to prove STARK-based accelerators for [Plonky2](https://github.com/mir-protocol/plonky2) circuits.

Stark proofs and verification are implemented via [Starky](https://github.com/0xPolygonZero/plonky2/tree/main/starky). This repository contains a modified fork of Starky to enable the support a 1-round AIR with random challenges and using the StarkyX AIR constraints. All the cryptographic primitives are imported from the [Plonky2](https://github.com/mir-protocol/plonky2) proving system.

## Building

StarkyX requires the use of the nightly Rust toolchain. To use it by default, run the following command:

```bash
rustup override set nightly
```

In the root directory of the project.

We recommend running the tests using the `--release` flag, as they are quite slow otherwise.

```bash
cargo test --release
```

## Usage

## Building an AIR computation using StarkyX

## Creating a STARK proof for an AIR computation

## Integrating into a Plonky2 circuit

StarkyX starks can be integrated into a [Plonky2](https://github.com/mir-protocol/plonky2) circuit

## Audit

StarkyX has been audited by [KALOS](https://kalos.xyz). The audit report can be found [here](https://hackmd.io/qS36EcIASx6Gt_2uNwlK4A). This repo was formerly named Curta, in the audit report it is referenced as Curta.
