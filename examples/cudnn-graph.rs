// follow the example in https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html

use cudarc::cudnn::result::{backend_execute, create_handle};
use cudarc::cudnn::sys::cudnnDataType_t;

use cudarc::cudnn::sys;
use cudarc::cudnn::{
    BackendTensorDescriptor, BackendTensorDescriptorBuilder, EngineConfigDescriptorBuilder,
    EngineDescriptorBuilder, ExecutionPlanDescriptorBuilder, OperationGraphDescriptorBuilder,
    OperationPointwiseDescriptorBuilder, PointwiseDescriptorBuilder, VariantPackDescriptorBuilder,
};
use cudarc::driver::CudaDevice;

fn create_tensor_descriptor(
    dim: &[i64],
    strides: &[i64],
    id: i64,
    alignment: i64,
    data_type: cudnnDataType_t,
) -> Result<BackendTensorDescriptor, cudarc::cudnn::CudnnError> {
    let tensor = BackendTensorDescriptorBuilder::new()?
        .set_tensor_data_type(&data_type)?
        .set_tensor_dimensions(dim)?
        .set_tensor_byte_alignment(alignment)?
        .set_tensor_strides(strides)?
        .set_tensor_unique_id(id)?
        .finalize()?;
    Ok(tensor)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 2;
    let x_dim = [1, n];
    let x_str = [n, 1];
    let alignment = 4i64;
    let data_type = cudnnDataType_t::CUDNN_DATA_FLOAT;

    let x_desc = create_tensor_descriptor(&x_dim, &x_str, 1, alignment, data_type)?;
    let y_desc = create_tensor_descriptor(&x_dim, &x_str, 2, alignment, data_type)?;
    let z_desc = create_tensor_descriptor(&x_dim, &x_str, 3, alignment, data_type)?;

    let add_desc = PointwiseDescriptorBuilder::new()?
        .set_mode(sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_ADD)?
        .finalize()?;
    let f_desc = OperationPointwiseDescriptorBuilder::new()?
        .set_pw_descriptor(&add_desc)?
        .set_x_descriptor(&x_desc)?
        .set_y_descriptor(&z_desc)?
        .set_b_descriptor(&y_desc)?
        .finalize()?;

    let handle = create_handle()?;

    let graph_desc = OperationGraphDescriptorBuilder::new()?
        .set_ops_pointwise(&f_desc)?
        .set_handle(&handle)?
        .finalize()?;
    let workspace_size = graph_desc.get_workspace_size()?;
    dbg!(workspace_size);

    let engine_desc = EngineDescriptorBuilder::new()?
        .set_operation_graph(&graph_desc)?
        .set_global_index(1)?
        .finalize()?;
    let engcfg = EngineConfigDescriptorBuilder::new()?
        .set_engine(&engine_desc)?
        .finalize()?;

    let plan = ExecutionPlanDescriptorBuilder::new()?
        .set_handle(&handle)?
        .set_engine_config(&engcfg)?
        .finalize()?;
    let workspace_size = plan.get_workspace_size()?;
    dbg!(workspace_size);

    // Allocate CUDA memory
    let uids = [1, 2, 3];
    let dev = CudaDevice::new(0)?;
    let mut b = dev.alloc_zeros::<f32>(2)?;
    let mut c = dev.alloc_zeros::<f32>(2)?;
    let mut d = dev.alloc_zeros::<f32>(2)?;
    let mut workspace = dev.alloc_zeros::<f32>(2)?;
    dev.htod_copy_into(vec![3.0; 2], &mut b)?;
    dev.htod_copy_into(vec![2.0; 2], &mut c)?;
    dev.htod_copy_into(vec![0.0; 2], &mut d)?;
    dev.htod_copy_into(vec![0.0; 2], &mut workspace)?;

    let mut dev_ptrs = [&mut b, &mut c, &mut d];
    let variant_pack_desc = VariantPackDescriptorBuilder::new()?
        .set_unique_ids(&uids)?
        .set_workspace(&mut workspace)?
        .set_data_pointers(&mut dev_ptrs)?
        .finalize()?;

    // run the graph
    backend_execute(handle, plan.descriptor, variant_pack_desc.descriptor)?;
    let mut d_host = vec![0.0; 2];
    dev.dtoh_sync_copy_into(&d, &mut d_host)?;
    dbg!(&d_host);

    // run the graph again
    dev.htod_copy_into(vec![4.0; 2], &mut b)?;
    backend_execute(handle, plan.descriptor, variant_pack_desc.descriptor)?;
    dev.dtoh_sync_copy_into(&d, &mut d_host)?;
    dbg!(d_host);

    Ok(())
}
