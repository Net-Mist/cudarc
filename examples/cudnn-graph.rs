// follow the example in https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html

use std::ffi::c_uint;
use std::ptr;

use cudarc::cudnn::result::{backend_finalize, backend_set_attribute, create_backend_descriptor};
use cudarc::cudnn::sys::{
    cudnnBackendAttributeName_t, cudnnBackendAttributeType_t, cudnnBackendDescriptorType_t,
    cudnnBackendDescriptor_t, cudnnDataType_t,
};

use cudarc::cudnn::sys::{self, lib};

fn main() -> Result<(), cudarc::cudnn::CudnnError> {
    let n = 1;
    let g = 1;
    let c = 3;
    let d = 1;
    let h = 224;
    let w = 224;

    let x_dim = [n, g, c, d, h, w];
    let x_str = [g * c * d * h * w, c * d * h * w, d * h * w, h * w, w, 1];
    let x_ui = 'x';
    let alignment = 4i64;
    let data_type = cudnnDataType_t::CUDNN_DATA_FLOAT;
    // let data_type = 0;

    unsafe {
        let desc: cudnnBackendDescriptor_t = create_backend_descriptor(
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR,
        )?;

        let array = [data_type];
        backend_set_attribute(
            desc,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DATA_TYPE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
            1,
            &(data_type) as *const cudnnDataType_t as *const std::ffi::c_void,
        )?;
        backend_set_attribute(
            desc,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DIMENSIONS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            6,
            x_dim.as_ptr() as *const std::ffi::c_void,
        )?;
        backend_set_attribute(
            desc,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_STRIDES,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            6,
            x_str.as_ptr() as *const std::ffi::c_void,
        )?;

        backend_set_attribute(
            desc,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_UNIQUE_ID,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            1,
            &(x_ui) as *const char as *const std::ffi::c_void,
        )?;

        backend_set_attribute(
            desc,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            1,
            &(alignment) as *const i64 as *const std::ffi::c_void,
        )?;

        backend_finalize(desc)?;
        Ok(())
    }

    // cudnnBackendDescriptorType_t.CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR
    // cudnnBackendDescriptorType_t.CUDNN_BACKEND_POINTWISE_DESCRIPTOR
    // cudnnPointwiseMode_t.CUDNN_POINTWISE_ADD
    // cudnnPointwiseMode_t.CUDNN_POINTWISE_RELU_FWD
}
