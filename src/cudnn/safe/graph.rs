use crate::cudnn::{result::{backend_create_descriptor, backend_finalize, backend_set_attribute}, sys::{cudnnBackendAttributeName_t, cudnnBackendAttributeType_t, cudnnBackendDescriptorType_t, cudnnBackendDescriptor_t, cudnnDataType_t}};

use super::CudnnError;

pub struct BackendTensorDescriptor {
    descriptor: cudnnBackendDescriptor_t,
}

impl BackendTensorDescriptor {
    pub fn new() -> Result<Self, CudnnError> {
        let descriptor: cudnnBackendDescriptor_t = backend_create_descriptor(
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR,
        )?;
        Ok(BackendTensorDescriptor {descriptor})
    }

    pub fn set_tensor_data_type(self, data_type: &cudnnDataType_t) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DATA_TYPE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
            1,
            data_type as *const cudnnDataType_t as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_tensor_dimensions(self, dim: &[i64]) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DIMENSIONS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            dim.len() as i64,
            dim.as_ptr() as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_tensor_strides(self, strides: &[i64]) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_STRIDES,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            strides.len() as i64,
            strides.as_ptr() as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_tensor_byte_alignment(self, alignment: i64) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            1,
            &(alignment) as *const i64 as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn finalize(self) -> Result<Self, CudnnError> {
        backend_finalize(self.descriptor)?;
        Ok(self)
    }

}
