# Curta

<a title="Rama, CC BY-SA 3.0 FR &lt;https://creativecommons.org/licenses/by-sa/3.0/fr/deed.en&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Curta_type_I-CnAM_40092-IMG_6721-white.jpg"><img width="256" alt="Curta type I-CnAM 40092-IMG 6721-white" src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Curta_type_I-CnAM_40092-IMG_6721-white.jpg/256px-Curta_type_I-CnAM_40092-IMG_6721-white.jpg"></a>


## Introduction


## Building
Curta requires the use of the nightly Rust toolchain. To use it by default, run the following command:

```bash
rustup override set nightly
```
In the root directory of the project.


We recommend running the tests using the `--release` flag, as they are quite slow otherwise.
```bash
cargo test --release
```

## Usage

## Building an AIR computation using Curta



## Integrating into a Plonky2 circuit
Curta starks can be integrated into a [Plonky2](https://github.com/mir-protocol/plonky2) circuit