# Wasm-Standalone Builder

This repository allows you to compile ONNX-formatted machine learning models into standalone Wasm binaries. ResNet 50 and MobileNetV2 are currently supported.

## Usage

Run `docker-compose` as below:

```shell
docker-compose build wasm-standalone-builder
docker-compose run wasm-standalone-builder
```

Then you'll see Wasm binaries in the `build``/` directory.

## Author

Kentaro Kuribayashi
