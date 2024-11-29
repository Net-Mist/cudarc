use core::mem::MaybeUninit;

use crate::{
    cudnn::{
        result::{
            backend_create_descriptor, backend_finalize, backend_get_attribute,
            backend_set_attribute,
        },
        sys::{
            cudnnBackendAttributeName_t, cudnnBackendAttributeType_t, cudnnBackendDescriptorType_t,
            cudnnBackendDescriptor_t, cudnnDataType_t, cudnnHandle_t, cudnnPointwiseMode_t,
        },
    },
    driver::{CudaSlice, DevicePtr, DevicePtrMut},
};

use super::{Cudnn, CudnnError};

pub struct BackendTensorDescriptorBuilder {
    descriptor: cudnnBackendDescriptor_t,
}

pub struct BackendTensorDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl BackendTensorDescriptorBuilder {
    pub fn new() -> Result<Self, CudnnError> {
        let descriptor: cudnnBackendDescriptor_t = backend_create_descriptor(
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR,
        )?;
        Ok(BackendTensorDescriptorBuilder { descriptor })
    }

    pub fn set_tensor_unique_id(self, id: i64) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_UNIQUE_ID,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            1,
            &id as *const i64 as *const std::ffi::c_void,
        )?;
        Ok(self)
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

    pub fn finalize(self) -> Result<BackendTensorDescriptor, CudnnError> {
        backend_finalize(self.descriptor)?;
        Ok(BackendTensorDescriptor {
            descriptor: self.descriptor,
        })
    }
}

pub struct PointwiseDescriptorBuilder {
    descriptor: cudnnBackendDescriptor_t,
}

pub struct PointwiseDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl PointwiseDescriptorBuilder {
    pub fn new() -> Result<Self, CudnnError> {
        let descriptor: cudnnBackendDescriptor_t = backend_create_descriptor(
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
        )?;
        Ok(PointwiseDescriptorBuilder { descriptor })
    }

    pub fn set_mode(self, mode: cudnnPointwiseMode_t) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_MODE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_POINTWISE_MODE,
            1,
            &(mode) as *const cudnnPointwiseMode_t as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn finalize(self) -> Result<PointwiseDescriptor, CudnnError> {
        backend_finalize(self.descriptor)?;
        Ok(PointwiseDescriptor {
            descriptor: self.descriptor,
        })
    }
}

pub struct OperationPointwiseDescriptorBuilder {
    descriptor: cudnnBackendDescriptor_t,
}

pub struct OperationPointwiseDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl OperationPointwiseDescriptorBuilder {
    pub fn new() -> Result<Self, CudnnError> {
        let descriptor: cudnnBackendDescriptor_t = backend_create_descriptor(
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
        )?;
        Ok(OperationPointwiseDescriptorBuilder { descriptor })
    }

    pub fn set_pw_descriptor(
        self,
        op_descriptor: &PointwiseDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(op_descriptor.descriptor) as *const cudnnBackendDescriptor_t
                as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_x_descriptor(
        self,
        x_descriptor: &BackendTensorDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(x_descriptor.descriptor) as *const cudnnBackendDescriptor_t
                as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_y_descriptor(
        self,
        y_descriptor: &BackendTensorDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(y_descriptor.descriptor) as *const cudnnBackendDescriptor_t
                as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_b_descriptor(
        self,
        b_descriptor: &BackendTensorDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(b_descriptor.descriptor) as *const cudnnBackendDescriptor_t
                as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn finalize(self) -> Result<OperationPointwiseDescriptor, CudnnError> {
        backend_finalize(self.descriptor)?;
        Ok(OperationPointwiseDescriptor {
            descriptor: self.descriptor,
        })
    }
}

pub struct OperationGraphDescriptorBuilder {
    descriptor: cudnnBackendDescriptor_t,
}

pub struct OperationGraphDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl OperationGraphDescriptorBuilder {
    pub fn new() -> Result<Self, CudnnError> {
        let descriptor: cudnnBackendDescriptor_t = backend_create_descriptor(
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
        )?;
        Ok(OperationGraphDescriptorBuilder { descriptor })
    }

    pub fn set_ops_pointwise(
        self,
        fprop: &OperationPointwiseDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_OPS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(fprop.descriptor) as *const _ as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_handle(self, handle: &cudnnHandle_t) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_HANDLE,
            1,
            handle as *const _ as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn finalize(self) -> Result<OperationGraphDescriptor, CudnnError> {
        backend_finalize(self.descriptor)?;
        Ok(OperationGraphDescriptor {
            descriptor: self.descriptor,
        })
    }
}

impl OperationGraphDescriptor {
    pub fn get_workspace_size(&self) -> Result<i64, CudnnError> {
        let mut workspace_size: MaybeUninit<i64> = MaybeUninit::uninit();
        let mut element_count: MaybeUninit<i64> = MaybeUninit::uninit();
        backend_get_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            1,
            element_count.as_mut_ptr(),
            workspace_size.as_mut_ptr() as *mut std::ffi::c_void,
        )?;
        Ok(unsafe { workspace_size.assume_init() })
    }
}

pub struct EngineDescriptorBuilder {
    descriptor: cudnnBackendDescriptor_t,
}

pub struct EngineDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl EngineDescriptorBuilder {
    pub fn new() -> Result<Self, CudnnError> {
        let descriptor: cudnnBackendDescriptor_t = backend_create_descriptor(
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINE_DESCRIPTOR,
        )?;
        Ok(EngineDescriptorBuilder { descriptor })
    }

    pub fn set_operation_graph(
        self,
        graph_desc: &OperationGraphDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(graph_desc.descriptor) as *const cudnnBackendDescriptor_t as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_global_index(self, gidx: i64) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            1,
            &gidx as *const i64 as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn finalize(self) -> Result<EngineDescriptor, CudnnError> {
        backend_finalize(self.descriptor)?;
        Ok(EngineDescriptor {
            descriptor: self.descriptor,
        })
    }
}

pub struct EngineConfigDescriptorBuilder {
    descriptor: cudnnBackendDescriptor_t,
}

pub struct EngineConfigDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl EngineConfigDescriptorBuilder {
    pub fn new() -> Result<Self, CudnnError> {
        let descriptor: cudnnBackendDescriptor_t = backend_create_descriptor(
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
        )?;
        Ok(EngineConfigDescriptorBuilder { descriptor })
    }

    pub fn set_engine(self, engine_desc: &EngineDescriptor) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_ENGINE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(engine_desc.descriptor) as *const cudnnBackendDescriptor_t as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn finalize(self) -> Result<EngineConfigDescriptor, CudnnError> {
        backend_finalize(self.descriptor)?;
        Ok(EngineConfigDescriptor {
            descriptor: self.descriptor,
        })
    }
}

pub struct ExecutionPlanDescriptorBuilder {
    descriptor: cudnnBackendDescriptor_t,
}

pub struct ExecutionPlanDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl ExecutionPlanDescriptorBuilder {
    pub fn new() -> Result<Self, CudnnError> {
        let descriptor: cudnnBackendDescriptor_t = backend_create_descriptor(
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
        )?;
        Ok(ExecutionPlanDescriptorBuilder { descriptor })
    }

    pub fn set_handle(mut self, handle: &cudnnHandle_t) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_HANDLE,
            1,
            handle as *const _ as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_engine_config(
        mut self,
        engcfg: &EngineConfigDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(engcfg.descriptor) as *const cudnnBackendDescriptor_t as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn finalize(self) -> Result<ExecutionPlanDescriptor, CudnnError> {
        backend_finalize(self.descriptor)?;
        Ok(ExecutionPlanDescriptor {
            descriptor: self.descriptor,
        })
    }
}

impl ExecutionPlanDescriptor {
    pub fn get_workspace_size(&self) -> Result<i64, CudnnError> {
        let mut workspace_size: MaybeUninit<i64> = MaybeUninit::uninit();
        let mut element_count: MaybeUninit<i64> = MaybeUninit::uninit();
        backend_get_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            1,
            element_count.as_mut_ptr(),
            workspace_size.as_mut_ptr() as *mut std::ffi::c_void,
        )?;
        Ok(unsafe { workspace_size.assume_init() })
    }
}

pub struct VariantPackDescriptorBuilder {
    descriptor: cudnnBackendDescriptor_t,
}

pub struct VariantPackDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl VariantPackDescriptorBuilder {
    pub fn new() -> Result<Self, CudnnError> {
        let descriptor: cudnnBackendDescriptor_t = backend_create_descriptor(
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
        )?;
        Ok(VariantPackDescriptorBuilder { descriptor })
    }

    pub fn set_unique_ids(self, uids: &[i64]) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            uids.len() as i64,
            uids.as_ptr() as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_workspace<T>(self, workspace: &mut CudaSlice<T>) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR,
            1,
            workspace.device_ptr_mut() as *mut _ as *mut std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_data_pointers<T>(
        self,
        data_pointers: &mut [&mut CudaSlice<T>],
    ) -> Result<Self, CudnnError> {
        let mut p = data_pointers
            .into_iter()
            .map(|x| *x.device_ptr_mut() as *mut std::ffi::c_void)
            .collect::<Vec<_>>();

        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR,
            data_pointers.len() as i64,
            p.as_mut_ptr() as *mut std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn finalize(self) -> Result<VariantPackDescriptor, CudnnError> {
        backend_finalize(self.descriptor)?;
        Ok(VariantPackDescriptor {
            descriptor: self.descriptor,
        })
    }
}
