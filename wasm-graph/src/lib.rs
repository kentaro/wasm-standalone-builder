/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#[macro_use]
extern crate lazy_static;

use std::mem;
use std::slice;

mod types;
mod utils;

use std::{collections::HashMap, convert::TryFrom, sync::Mutex};

use ndarray::Array;
use tvm_graph_rt::{Graph, GraphExecutor, SystemLibModule, Tensor as TVMTensor};

use types::*;

use serde::{Deserialize, Serialize};

use image::{ImageBuffer, Rgb};
use image::imageops::FilterType;

const IMG_HEIGHT: usize = 224;
const IMG_WIDTH: usize = 224;

#[derive(Serialize, Deserialize)]
struct Result {
    data: Vec<f32>,
}

extern "C" {
    fn __wasm_call_ctors();
}

lazy_static! {
    static ref SYSLIB: SystemLibModule = SystemLibModule::default();
    static ref GRAPH_EXECUTOR: Mutex<GraphExecutor<'static, 'static>> = {
        unsafe {
            // This is necessary to invoke TVMBackendRegisterSystemLibSymbol
            // API calls.
            __wasm_call_ctors();
        }
        let graph = Graph::try_from(
            include_str!(
                "/workspaces/tvm/apps/wasm-standalone/wasm-graph/lib/graph.json"
            )
        ).unwrap();

        let params_bytes =
            include_bytes!(
                "/workspaces/tvm/apps/wasm-standalone/wasm-graph/lib/graph.params"
            );
        let params = tvm_graph_rt::load_param_dict(params_bytes)
            .unwrap()
            .into_iter()
            .map(|(k, v)| (k, v.to_owned()))
            .collect::<HashMap<String, TVMTensor<'static>>>();

        let mut exec = GraphExecutor::new(graph, &*SYSLIB).unwrap();

        exec.load_params(params);

        Mutex::new(exec)
    };
}

/// # Safety
#[no_mangle]
pub unsafe extern "C" fn predict(index: *const u8, length: usize) -> i32 {
    let slice = unsafe { slice::from_raw_parts(index, length) };
    let img_buffer = image::RgbImage::from_raw(224, 224, slice.to_vec()).unwrap();
    //let img = image::DynamicImage::ImageRgb8(img_buffer);
    let img_tensor = data_preprocess(img_buffer);
    let input = img_tensor.as_dltensor().into();

    // since this executor is not multi-threaded, we can acquire lock once
    let mut executor = GRAPH_EXECUTOR.lock().unwrap();
    executor.set_input("data", input);
    executor.run();

    let output = executor.get_output(0).unwrap().as_dltensor(false);
    let out_tensor: Tensor = output.into();
    //let output = out_tensor.to_vec::<f32>();
    let output = out_tensor;

    let out_size = unsafe { utils::store_output(index as i32, output) };
    out_size as i32

    // let result = Result {
    //   data: out_tensor.to_vec::<f32>(),
    // };

    // let json = serde_json::to_string(&result).unwrap();
    // json.as_ptr()
}

/// https://github.com/apache/tvm/blob/main/apps/wasm-standalone/wasm-runtime/tests/test_graph_resnet50/src/main.rs#L92-L118
//fn data_preprocess(img: image::DynamicImage) -> Tensor {
fn data_preprocess(img: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Tensor {
    let resized = image::imageops::resize(&img, IMG_HEIGHT as u32, IMG_WIDTH as u32, FilterType::Nearest);
    let img = resized;//img.to_rgb32f();

    let mut pixels: Vec<f32> = vec![];
    for pixel in img.pixels() {
        // normalize the RGB channels using mean, std of imagenet1k
        let tmp = [
            (pixel[0] as f32 - 123.0) / 58.395, // R
            (pixel[1] as f32 - 117.0) / 57.12,  // G
            (pixel[2] as f32 - 104.0) / 57.375, // B
        ];
        for e in &tmp {
            pixels.push(*e);
        }
    }

    // (H,W,C) -> (C,H,W)
    let arr = Array::from_shape_vec((IMG_HEIGHT, IMG_WIDTH, 3), pixels).unwrap();
    let arr = arr.permuted_axes([2, 0, 1]);
    let arr = Array::from_iter(arr.into_iter().copied());

    Tensor::from(arr)
}
