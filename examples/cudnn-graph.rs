// follow the example in https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html

use std::ffi::c_uint;
use std::mem::MaybeUninit;
use std::ptr;

use cudarc::cudnn::result::{
    backend_create_descriptor, backend_finalize, backend_get_attribute, backend_set_attribute,
    create_handle,
};
use cudarc::cudnn::sys::{
    cudnnBackendAttributeName_t, cudnnBackendAttributeType_t, cudnnBackendDescriptorType_t,
    cudnnBackendDescriptor_t, cudnnDataType_t, cudnnHandle_t,
};

use cudarc::cudnn::sys::{self, lib};
use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};

fn create_tensor_descriptor(
    dim: &[i64],
    strides: &[i64],
    id: i64,
    alignment: i64,
    data_type: cudnnDataType_t,
) -> Result<cudnnBackendDescriptor_t, cudarc::cudnn::CudnnError> {
    let desc: cudnnBackendDescriptor_t =
        backend_create_descriptor(cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR)?;

    // let array = [data_type];
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
        dim.len() as i64,
        dim.as_ptr() as *const std::ffi::c_void,
    )?;
    backend_set_attribute(
        desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_STRIDES,
        cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        strides.len() as i64,
        strides.as_ptr() as *const std::ffi::c_void,
    )?;

    backend_set_attribute(
        desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_UNIQUE_ID,
        cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        1,
        &(id) as *const _ as *const std::ffi::c_void,
    )?;

    backend_set_attribute(
        desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
        cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        1,
        &(alignment) as *const i64 as *const std::ffi::c_void,
    )?;
    println!("finalize");
    backend_finalize(desc)?;
    println!("done");
    Ok(desc)
}

fn create_add_descriptor() -> Result<cudnnBackendDescriptor_t, cudarc::cudnn::CudnnError> {
    // attributes are
    // CUDNN_ATTR_POINTWISE_MODE
    // CUDNN_ATTR_POINTWISE_MATH_PREC
    // CUDNN_ATTR_POINTWISE_NAN_PROPAGATION
    // CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP
    // CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP
    // CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE
    // CUDNN_ATTR_POINTWISE_ELU_ALPHA
    // CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA
    // CUDNN_ATTR_POINTWISE_SWISH_BETA
    let desc: cudnnBackendDescriptor_t = backend_create_descriptor(
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
    )?;
    let mode = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_ADD;
    backend_set_attribute(
        desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_POINTWISE_MODE,
        cudnnBackendAttributeType_t::CUDNN_TYPE_POINTWISE_MODE,
        1,
        &(mode) as *const sys::cudnnPointwiseMode_t as *const std::ffi::c_void,
    )?;

    backend_finalize(desc)?;
    Ok(desc)
}

/// cuda doc : https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-graph-library.html#cudnn-backend-operation-pointwise-descriptor
/// Represents a pointwise operation that implements one of the following equation:
/// * Y = op(alpha1 * X)
/// * Y = op(alpha1 * X, alpha2 * B)
fn create_pointwise_operation_descriptor(
    op_descriptor: cudnnBackendDescriptor_t,
    x_descriptor: cudnnBackendDescriptor_t,
    y_descriptor: cudnnBackendDescriptor_t,
    b_descriptor: Option<cudnnBackendDescriptor_t>,
) -> Result<cudnnBackendDescriptor_t, cudarc::cudnn::CudnnError> {
    let desc: cudnnBackendDescriptor_t = backend_create_descriptor(
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
    )?;
    backend_set_attribute(
        desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
        cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
        1,
        &(op_descriptor) as *const _ as *const std::ffi::c_void,
    )?;
    backend_set_attribute(
        desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
        cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
        1,
        &(x_descriptor) as *const _ as *const std::ffi::c_void,
    )?;
    backend_set_attribute(
        desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
        cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
        1,
        &(y_descriptor) as *const _ as *const std::ffi::c_void,
    )?;
    if let Some(b_descriptor) = b_descriptor {
        backend_set_attribute(
            desc,
            cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
            cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(b_descriptor) as *const _ as *const std::ffi::c_void,
        )?;
    }
    backend_finalize(desc)?;
    Ok(desc)
}

fn create_operation_graph_descriptor(
    fprop: &cudnnBackendDescriptor_t,
    handle: &cudnnHandle_t,
) -> Result<cudnnBackendDescriptor_t, cudarc::cudnn::CudnnError> {
    let desc: cudnnBackendDescriptor_t = backend_create_descriptor(
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
    )
    .unwrap();

    backend_set_attribute(
        desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_OPS,
        cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
        1,
        fprop as *const _ as *const std::ffi::c_void,
    )?;
    backend_set_attribute(
        desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
        cudnnBackendAttributeType_t::CUDNN_TYPE_HANDLE,
        1,
        handle as *const _ as *const std::ffi::c_void,
    )?;

    backend_finalize(desc)?;
    Ok(desc)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 2;

    let x_dim = [1, n];
    let x_str = [n, 1];
    let x_ui = 1;
    let alignment = 4i64;
    let data_type = cudnnDataType_t::CUDNN_DATA_FLOAT;

    println!("create tensor descriptor");

    let x_desc = create_tensor_descriptor(&x_dim, &x_str, x_ui, alignment, data_type)?;

    println!("x done");

    let y_desc = create_tensor_descriptor(&x_dim, &x_str, 2, alignment, data_type)?;

    let z_desc = create_tensor_descriptor(&x_dim, &x_str, 3, alignment, data_type)?;

    println!("create add");

    let add_desc = create_add_descriptor()?;

    let f_desc = create_pointwise_operation_descriptor(add_desc, x_desc, z_desc, Some(y_desc))?;

    let handle = create_handle()?;
    let graph_desc = create_operation_graph_descriptor(&f_desc, &handle)?;

    let mut workspace_size: MaybeUninit<i64> = MaybeUninit::uninit();
    let mut element_count: MaybeUninit<i64> = MaybeUninit::uninit();
    backend_get_attribute(
        graph_desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
        cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        1,
        element_count.as_mut_ptr(),
        workspace_size.as_mut_ptr() as *mut std::ffi::c_void,
    )?;
    let (v, w) = unsafe { (workspace_size.assume_init(), element_count.assume_init()) };
    dbg!(v);
    dbg!(w);

    // engine config
    let engine_desc =
        backend_create_descriptor(cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINE_DESCRIPTOR)?;
    backend_set_attribute(
        engine_desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
        cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
        1,
        &graph_desc as *const _ as *const std::ffi::c_void,
    )?;
    let gidx: i64 = 1;
    backend_set_attribute(
        engine_desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
        cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        1,
        &gidx as *const i64 as *const std::ffi::c_void,
    )?;
    backend_finalize(engine_desc)?;

    println!("engine config done");

    let engcfg = backend_create_descriptor(
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
    )?;
    backend_set_attribute(
        engcfg,
        cudnnBackendAttributeName_t::CUDNN_ATTR_ENGINECFG_ENGINE,
        cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
        1,
        &engine_desc as *const _ as *const std::ffi::c_void,
    )?;

    println!("ploooooooooooop");
    backend_finalize(engcfg)?;

    // execution plan
    let plan = backend_create_descriptor(
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
    )?;
    backend_set_attribute(
        plan,
        cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
        cudnnBackendAttributeType_t::CUDNN_TYPE_HANDLE,
        1,
        &handle as *const _ as *const std::ffi::c_void,
    )?;
    backend_set_attribute(
        plan,
        cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
        cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR,
        1,
        &engcfg as *const _ as *const std::ffi::c_void,
    )?;
    backend_finalize(plan)?;

    let mut workspace_size: MaybeUninit<i64> = MaybeUninit::uninit();
    let mut element_count: MaybeUninit<i64> = MaybeUninit::uninit();

    let (workspace_size, element_count) = unsafe {
        lib()
            .cudnnBackendGetAttribute(
                plan,
                cudnnBackendAttributeName_t::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
                1,
                element_count.as_mut_ptr(),
                workspace_size.as_mut_ptr() as *mut std::ffi::c_void,
            )
            .result()?;
        (workspace_size.assume_init(), element_count.assume_init())
    };
    dbg!(workspace_size);
    dbg!(element_count);

    // variant pack descriptor
    let variant_pack_desc = backend_create_descriptor(
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
    )?;

    let uids = [1i64, 2, 3];
    backend_set_attribute(
        variant_pack_desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
        cudnnBackendAttributeType_t::CUDNN_TYPE_INT64,
        3,
        uids.as_ptr() as *const std::ffi::c_void,
    )?;

    let dev = CudaDevice::new(0)?;
    let mut b = dev.alloc_zeros::<f32>(2)?;
    dev.htod_copy_into(vec![3.0; 2], &mut b)?;
    let mut c = dev.alloc_zeros::<f32>(2)?;
    dev.htod_copy_into(vec![2.0; 2], &mut c)?;
    let mut d = dev.alloc_zeros::<f32>(2)?;
    dev.htod_copy_into(vec![0.0; 2], &mut d)?;
    let mut workspace = dev.alloc_zeros::<f32>(2)?;
    dev.htod_copy_into(vec![0.0; 2], &mut workspace)?;

    let mut dev_ptrs = [
        *b.device_ptr_mut() as *mut std::ffi::c_void,
        *c.device_ptr_mut() as *mut std::ffi::c_void,
        *d.device_ptr_mut() as *mut std::ffi::c_void,
    ];

    backend_set_attribute(
        variant_pack_desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
        cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR,
        1,
        &workspace as *const _ as *const std::ffi::c_void,
    )?;

    backend_set_attribute(
        variant_pack_desc,
        cudnnBackendAttributeName_t::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
        cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR,
        3,
        dev_ptrs.as_mut_ptr() as *mut std::ffi::c_void,
    )?;

    backend_finalize(variant_pack_desc)?;

    unsafe {
        lib().cudnnBackendExecute(handle, plan, variant_pack_desc);
    }


    println!("done");
    let d_host = dev.sync_reclaim(d)?;
    dbg!(d_host);

    Ok(())
}
