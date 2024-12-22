use core::mem::MaybeUninit;

use crate::{
    cudnn::{
        result::{
            backend_create_descriptor, backend_finalize, backend_get_attribute,
            backend_set_attribute,
        },
        sys::{
            cudnnBackendAttributeName_t, cudnnBackendAttributeType_t, cudnnBackendDescriptorType_t,
            cudnnBackendDescriptor_t, cudnnBackendHeurMode_t, cudnnBackendKnobType_t,
            cudnnBackendNumericalNote_t, cudnnDataType_t, cudnnHandle_t, cudnnPointwiseMode_t,
        },
    },
    driver::{CudaSlice, DevicePtr, DevicePtrMut},
};

use super::{Cudnn, CudnnError};

pub trait Descriptor: Sized {
    fn new() -> Result<Self, CudnnError>;
    fn from_descriptor(descriptor: cudnnBackendDescriptor_t) -> Self;
    fn get_descriptor(&self) -> cudnnBackendDescriptor_t;
}

pub trait Builder<T>: Sized {
    fn new() -> Result<Self, CudnnError>;
    fn finalize(self) -> Result<T, CudnnError>;
    fn get_descriptor(&self) -> cudnnBackendDescriptor_t;

    fn new_descriptor() -> Result<cudnnBackendDescriptor_t, CudnnError> {
        let descriptor = Self::new()?;
        Ok(descriptor.get_descriptor())
    }
}

macro_rules! impl_builder {
    ($builder:ident, $descriptor:ident, $backend_type:expr) => {
        impl Builder<$descriptor> for $builder {
            fn new() -> Result<Self, CudnnError> {
                let descriptor: cudnnBackendDescriptor_t =
                    backend_create_descriptor($backend_type)?;
                Ok($builder { descriptor })
            }

            fn finalize(self) -> Result<$descriptor, CudnnError> {
                backend_finalize(self.descriptor)?;
                Ok($descriptor {
                    descriptor: self.descriptor,
                })
            }

            fn get_descriptor(&self) -> cudnnBackendDescriptor_t {
                self.descriptor
            }
        }

        impl Descriptor for $descriptor {
            fn new() -> Result<Self, CudnnError> {
                let descriptor: cudnnBackendDescriptor_t =
                    backend_create_descriptor($backend_type)?;
                Ok($descriptor { descriptor })
            }

            fn from_descriptor(descriptor: cudnnBackendDescriptor_t) -> Self {
                $descriptor { descriptor }
            }

            fn get_descriptor(&self) -> cudnnBackendDescriptor_t {
                self.descriptor
            }
        }
    };
}

// Apply the macro to implement the Builder and Descriptor traits for various descriptor types
impl_builder!(
    BackendTensorDescriptorBuilder,
    BackendTensorDescriptor,
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR
);
impl_builder!(
    PointwiseDescriptorBuilder,
    PointwiseDescriptor,
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_POINTWISE_DESCRIPTOR
);
impl_builder!(
    OperationPointwiseDescriptorBuilder,
    OperationPointwiseDescriptor,
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR
);
impl_builder!(
    OperationGraphDescriptorBuilder,
    OperationGraphDescriptor,
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
);
impl_builder!(
    EngineDescriptorBuilder,
    EngineDescriptor,
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINE_DESCRIPTOR
);
impl_builder!(
    EngineConfigDescriptorBuilder,
    EngineConfigDescriptor,
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR
);
impl_builder!(
    ExecutionPlanDescriptorBuilder,
    ExecutionPlanDescriptor,
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR
);
impl_builder!(
    VariantPackDescriptorBuilder,
    VariantPackDescriptor,
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR
);
impl_builder!(
    EngineHeuristicsDescriptorBuilder,
    EngineHeuristicsDescriptor,
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR
);
impl_builder!(
    KnobChoiceDescriptorBuilder,
    KnobChoiceDescriptor,
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR
);
impl_builder!(
    MatmulDescriptorBuilder,
    MatmulDescriptor,
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_MATMUL_DESCRIPTOR
);
impl_builder!(
    OperationMatmulDescriptorBuilder,
    OperationMatmulDescriptor,
    cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR
);

fn get_attributes<T: Descriptor, U: Builder<T>>(
    descriptor: cudnnBackendDescriptor_t,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    max_count: i64,
) -> Result<Vec<T>, CudnnError> {
    let mut count = MaybeUninit::uninit();
    let mut descriptors = Vec::with_capacity(max_count as usize);
    for _ in 0..max_count {
        descriptors.push(U::new_descriptor()?);
    }

    backend_get_attribute(
        descriptor,
        attribute_name,
        attribute_type,
        max_count,
        count.as_mut_ptr(),
        descriptors.as_mut_ptr() as *mut std::ffi::c_void,
    )?;
    let count_i64 = unsafe { count.assume_init() };
    let results = descriptors
        .drain(..count_i64 as usize)
        .into_iter()
        .map(|desc| T::from_descriptor(desc))
        .collect();

    Ok(results)
}

fn get_attribute<T: Descriptor, U: Builder<T>>(
    descriptor: cudnnBackendDescriptor_t,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
) -> Result<T, CudnnError> {
    let mut count = MaybeUninit::uninit();
    let mut descriptors = Vec::with_capacity(1 as usize);
    for _ in 0..1 {
        descriptors.push(U::new_descriptor()?);
    }

    backend_get_attribute(
        descriptor,
        attribute_name,
        attribute_type,
        1,
        count.as_mut_ptr(),
        descriptors.as_mut_ptr() as *mut std::ffi::c_void,
    )?;
    let count_i64 = unsafe { count.assume_init() };
    Ok(T::from_descriptor(descriptors[0]))
}
fn get_basic_attributes<T>(
    descriptor: cudnnBackendDescriptor_t,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    max_count: i64,
) -> Result<Vec<T>, CudnnError> {
    let mut count = MaybeUninit::uninit();
    let mut descriptors = Vec::with_capacity(max_count as usize);
    for _ in 0..max_count {
        descriptors.push(MaybeUninit::uninit());
    }

    backend_get_attribute(
        descriptor,
        attribute_name,
        attribute_type,
        max_count,
        count.as_mut_ptr(),
        descriptors.as_mut_ptr() as *mut std::ffi::c_void,
    )?;
    let count_i64 = unsafe { count.assume_init() };
    let results = descriptors
        .drain(..count_i64 as usize)
        .into_iter()
        .map(|desc| unsafe { desc.assume_init() })
        .collect();

    Ok(results)
}


fn get_basic_attribute<T>(
    descriptor: cudnnBackendDescriptor_t,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
) -> Result<T, CudnnError> {
    let mut count = MaybeUninit::uninit();
    let mut element = MaybeUninit::uninit();
    backend_get_attribute(
        descriptor,
        attribute_name,
        attribute_type,
        1,
        count.as_mut_ptr(),
        element.as_mut_ptr() as *mut std::ffi::c_void,
    )?;
    Ok(unsafe { element.assume_init() })
}

pub struct BackendTensorDescriptorBuilder {
    pub descriptor: cudnnBackendDescriptor_t,
}

pub struct BackendTensorDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl BackendTensorDescriptorBuilder {
    pub fn set_unique_id(self, id: i64) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_UNIQUE_ID,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            1,
            &id as *const i64 as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_data_type(self, data_type: &cudnnDataType_t) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DATA_TYPE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
            1,
            data_type as *const cudnnDataType_t as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_dimensions(self, dim: &[i64]) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_DIMENSIONS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            dim.len() as i64,
            dim.as_ptr() as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_strides(self, strides: &[i64]) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_STRIDES,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            strides.len() as i64,
            strides.as_ptr() as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_byte_alignment(self, alignment: i64) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            1,
            &(alignment) as *const i64 as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_is_virtual(self, is_virtual: bool) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_IS_VIRTUAL,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BOOLEAN,
            1,
            &(is_virtual) as *const bool as *const std::ffi::c_void,
        )?;
        Ok(self)
    }
}

pub struct PointwiseDescriptorBuilder {
    pub descriptor: cudnnBackendDescriptor_t,
}

pub struct PointwiseDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl PointwiseDescriptorBuilder {
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
}

pub struct OperationPointwiseDescriptorBuilder {
    pub descriptor: cudnnBackendDescriptor_t,
}

pub struct OperationPointwiseDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl OperationPointwiseDescriptorBuilder {
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

    pub fn set_math_prec(self, math_prec: &cudnnDataType_t) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_MATH_PREC,
            cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
            1,
            math_prec as *const cudnnDataType_t as *const std::ffi::c_void,
        )?;
        Ok(self)
    }
}

pub struct OperationGraphDescriptorBuilder {
    pub descriptor: cudnnBackendDescriptor_t,
}

pub struct OperationGraphDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl OperationGraphDescriptorBuilder {
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

    pub fn set_ops_matmul(self, matmul: &OperationMatmulDescriptor) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_OPS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(matmul.descriptor) as *const _ as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_ops(self, matmul: &OperationMatmulDescriptor, pointwises: &[&OperationPointwiseDescriptor]) -> Result<Self, CudnnError> {
        let mut elements = vec![matmul.descriptor]
            .into_iter()
            .chain(pointwises.iter().map(|x| x.descriptor))
            .collect::<Vec<_>>();
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_OPS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1 + pointwises.len() as i64,
            elements.as_mut_ptr() as *const std::ffi::c_void,
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
}

impl OperationGraphDescriptor {
    pub fn get_engine_glocal_count(&self) -> Result<i64, CudnnError> {
        get_basic_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        )
    }

}

pub struct EngineDescriptorBuilder {
    pub descriptor: cudnnBackendDescriptor_t,
}

pub struct EngineDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl EngineDescriptorBuilder {
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
}

impl EngineDescriptor {
    pub fn get_global_index(&self) -> Result<i64, CudnnError> {
        get_basic_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        )
    }

    pub fn get_numerical_note(&self) -> Result<Vec<cudnnBackendNumericalNote_t>, CudnnError> {
        get_basic_attributes(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_NUMERICAL_NOTE,
            10
        )
    }
}

pub struct EngineConfigDescriptorBuilder {
    pub descriptor: cudnnBackendDescriptor_t,
}

pub struct EngineConfigDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl EngineConfigDescriptorBuilder {
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
}

impl EngineConfigDescriptor {
    pub fn get_engine(&self) -> Result<EngineDescriptor, CudnnError> {
        get_attribute::<EngineDescriptor, EngineDescriptorBuilder>(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_ENGINE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
        )
    }

    pub fn get_knob_choices(
        &self,
        max_n_knobs: i64,
    ) -> Result<Vec<KnobChoiceDescriptor>, CudnnError> {
        get_attributes::<KnobChoiceDescriptor, KnobChoiceDescriptorBuilder>(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            max_n_knobs,
        )
    }
}

pub struct ExecutionPlanDescriptorBuilder {
    pub descriptor: cudnnBackendDescriptor_t,
}

pub struct ExecutionPlanDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl ExecutionPlanDescriptorBuilder {
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
}

impl ExecutionPlanDescriptor {
    pub fn get_workspace_size(&self) -> Result<i64, CudnnError> {
        get_basic_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        )
    }
}

pub struct VariantPackDescriptorBuilder {
    pub descriptor: cudnnBackendDescriptor_t,
}

pub struct VariantPackDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl VariantPackDescriptorBuilder {
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
}

pub struct EngineHeuristicsDescriptorBuilder {
    pub descriptor: cudnnBackendDescriptor_t,
}

pub struct EngineHeuristicsDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl EngineHeuristicsDescriptorBuilder {
    pub fn set_operation_graph(
        self,
        graph_desc: &OperationGraphDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(graph_desc.descriptor) as *const cudnnBackendDescriptor_t as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_mode(self, mode: &cudnnBackendHeurMode_t) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_MODE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_HEUR_MODE,
            1,
            mode as *const cudnnBackendHeurMode_t as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_sm_count_target(self, sm_count: i32) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT32,
            1,
            &sm_count as *const i32 as *const std::ffi::c_void,
        )?;
        Ok(self)
    }
}

impl EngineHeuristicsDescriptor {
    pub fn get_results(
        &self,
        max_n_results: i64,
    ) -> Result<Vec<EngineConfigDescriptor>, CudnnError> {
        get_attributes::<EngineConfigDescriptor, EngineConfigDescriptorBuilder>(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINEHEUR_RESULTS,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            max_n_results,
        )
    }
}

pub struct KnobChoiceDescriptorBuilder {
    pub descriptor: cudnnBackendDescriptor_t,
}

pub struct KnobChoiceDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl KnobChoiceDescriptorBuilder {
    pub fn set_knob_type(self, knob_type: cudnnBackendKnobType_t) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_KNOB_TYPE,
            1,
            &knob_type as *const cudnnBackendKnobType_t as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_knob_value(self, knob_value: i64) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            1,
            &knob_value as *const i64 as *const std::ffi::c_void,
        )?;
        Ok(self)
    }
}

impl KnobChoiceDescriptor {
    pub fn get_knob_type(&self) -> Result<cudnnBackendKnobType_t, CudnnError> {
        let mut knob_type: MaybeUninit<cudnnBackendKnobType_t> = MaybeUninit::uninit();
        backend_get_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_KNOB_TYPE,
            1,
            std::ptr::null_mut(),
            knob_type.as_mut_ptr() as *mut std::ffi::c_void,
        )?;
        Ok(unsafe { knob_type.assume_init() })
    }

    pub fn get_knob_value(&self) -> Result<i64, CudnnError> {
        let mut knob_value: MaybeUninit<i64> = MaybeUninit::uninit();
        backend_get_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            1,
            std::ptr::null_mut(),
            knob_value.as_mut_ptr() as *mut std::ffi::c_void,
        )?;
        Ok(unsafe { knob_value.assume_init() })
    }
}

pub struct MatmulDescriptorBuilder {
    pub descriptor: cudnnBackendDescriptor_t,
}

pub struct MatmulDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl MatmulDescriptorBuilder {
    pub fn new() -> Result<Self, CudnnError> {
        let descriptor: cudnnBackendDescriptor_t = backend_create_descriptor(
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_MATMUL_DESCRIPTOR,
        )?;
        Ok(MatmulDescriptorBuilder { descriptor })
    }

    pub fn set_compute_type(self, compute_type: &cudnnDataType_t) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_MATMUL_COMP_TYPE,
            cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE,
            1,
            compute_type as *const cudnnDataType_t as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn finalize(self) -> Result<MatmulDescriptor, CudnnError> {
        backend_finalize(self.descriptor)?;
        Ok(MatmulDescriptor {
            descriptor: self.descriptor,
        })
    }
}

pub struct OperationMatmulDescriptorBuilder {
    pub descriptor: cudnnBackendDescriptor_t,
}

pub struct OperationMatmulDescriptor {
    pub descriptor: cudnnBackendDescriptor_t,
}

impl OperationMatmulDescriptorBuilder {
    pub fn new() -> Result<Self, CudnnError> {
        let descriptor: cudnnBackendDescriptor_t = backend_create_descriptor(
            cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
        )?;
        Ok(OperationMatmulDescriptorBuilder { descriptor })
    }

    pub fn set_a_descriptor(
        self,
        a_descriptor: &BackendTensorDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_ADESC,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(a_descriptor.descriptor) as *const cudnnBackendDescriptor_t
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
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_BDESC,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(b_descriptor.descriptor) as *const cudnnBackendDescriptor_t
                as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_c_descriptor(
        self,
        c_descriptor: &BackendTensorDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_CDESC,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(c_descriptor.descriptor) as *const cudnnBackendDescriptor_t
                as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_irregularly_strided_batch_count(self, count: i64) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT,
            cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
            1,
            &count as *const i64 as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_gemm_m_override_desc(
        self,
        m_override_desc: &BackendTensorDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_override_desc.descriptor) as *const cudnnBackendDescriptor_t
                as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_gemm_n_override_desc(
        self,
        n_override_desc: &BackendTensorDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(n_override_desc.descriptor) as *const cudnnBackendDescriptor_t
                as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_gemm_k_override_desc(
        self,
        k_override_desc: &BackendTensorDescriptor,
    ) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(k_override_desc.descriptor) as *const cudnnBackendDescriptor_t
                as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn set_matmul_desc(self, matmul_desc: &MatmulDescriptor) -> Result<Self, CudnnError> {
        backend_set_attribute(
            self.descriptor,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_MATMUL_DESC,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(matmul_desc.descriptor) as *const cudnnBackendDescriptor_t as *const std::ffi::c_void,
        )?;
        Ok(self)
    }

    pub fn finalize(self) -> Result<OperationMatmulDescriptor, CudnnError> {
        backend_finalize(self.descriptor)?;
        Ok(OperationMatmulDescriptor {
            descriptor: self.descriptor,
        })
    }
}
