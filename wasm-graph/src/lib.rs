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
#[macro_use]
extern crate serde_derive;

use std::mem;
use std::slice;

mod types;
mod utils;

use std::{collections::HashMap, convert::TryFrom, sync::Mutex};

use tvm_graph_rt::{Graph, GraphExecutor, SystemLibModule, Tensor as TVMTensor};

use types::*;

use serde_json;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct ImageInfo {
    data: Vec<u8>,
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
                "/work/tvm/apps/wasm-standalone/wasm-graph/lib/graph.json"
            )
        ).unwrap();

        let params_bytes =
            include_bytes!(
                "/work/tvm/apps/wasm-standalone/wasm-graph/lib/graph.params"
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

#[no_mangle]
pub extern "C" fn predict(index: *const u8, length: usize) -> *const u8 {
    // let in_tensor = unsafe {
    //   utils::load_input(wasm_addr as i32, in_size * mem::size_of::<u64>() as usize)
    // };
    // let input: TVMTensor = in_tensor.as_dltensor().into();

    let slice = unsafe {
      slice::from_raw_parts(index, length * mem::size_of::<u64>() as usize)
    };
    let img = Tensor::new(
        DataType::INT8,
        vec![224, 224, 3],
        vec![],
        slice.to_vec(),
    );
    
    let input: TVMTensor = img.as_dltensor().into();

    // since this executor is not multi-threaded, we can acquire lock once
    let mut executor = GRAPH_EXECUTOR.lock().unwrap();
    executor.set_input("data", input);
    executor.run();

    // let output = executor.get_output(0).unwrap().as_dltensor(false);
    // let out_tensor: Tensor = output.into();
    let image_info = ImageInfo {
        data: vec![1, 2, 3, 4, 5],
    };

    let json = serde_json::to_string(&image_info).unwrap();
    json.as_ptr()
}
