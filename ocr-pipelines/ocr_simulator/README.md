# OCR Simulator

This folder is a stub/pointer to the OCR simulator library used to induce **realistic** OCR noise to texts, as described in:

> **ACL 2025**: *Cheap Character Noise for OCR-Robust Multilingual Embeddings*
> ([paper](https://aclanthology.org/2025.findings-acl.609/))

## Full Implementation

The complete `ocr_simulator` library is available in the canonical ACL 2025 repository:

> [impresso/ocr-robust-multilingual-embeddings/ocr_simulator](https://github.com/impresso/ocr-robust-multilingual-embeddings/tree/main/ocr_simulator)

## Description

Unlike the random character noise approach (see `generate_random_character_noise/`), the OCR simulator produces **realistic** character-level errors that mimic actual OCR pipeline outputs. It models confusions specific to font types (e.g. blackletter/Fraktur), scanning artefacts, and segmentation errors.

## Usage

Please refer to the canonical repository linked above for installation and usage instructions.
