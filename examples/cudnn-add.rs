// follow the example in https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/01_matmul_bias.ipynb

use cudarc::cudnn::result::{backend_execute, create_handle};
use cudarc::cudnn::sys::{cudnnBackendHeurMode_t, cudnnDataType_t};

use cudarc::cudnn::{sys, Builder, EngineHeuristicsDescriptorBuilder, MatmulDescriptorBuilder, OperationMatmulDescriptorBuilder};
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
        .set_data_type(&data_type)?
        .set_dimensions(dim)?
        .set_byte_alignment(alignment)?
        .set_strides(strides)?
        .set_unique_id(id)?
        .finalize()?;
    Ok(tensor)
}

fn create_virtual_tensor_descriptor(
    dim: &[i64],
    strides: &[i64],
    id: i64,
    alignment: i64,
    data_type: cudnnDataType_t,
) -> Result<BackendTensorDescriptor, cudarc::cudnn::CudnnError> {
    let tensor = BackendTensorDescriptorBuilder::new()?
        // .set_data_type(&data_type)?
        .set_dimensions(dim)?
        .set_byte_alignment(alignment)?
        .set_strides(strides)?
        .set_unique_id(id)?
        .set_is_virtual(true)?
        .finalize()?;
    Ok(tensor)
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batch = 16;
    let m = 128;
    let n = 128;
    let k = 512;
    let input_type = cudnnDataType_t::CUDNN_DATA_FLOAT; // TODO test FP16
    let intermediate_type = cudnnDataType_t::CUDNN_DATA_FLOAT;
    let compute_type = cudnnDataType_t::CUDNN_DATA_FLOAT; // TODO find the difference between them
    let alignment = 32i64; // TODO find the best value

    // TODO find the best strides
    let a_desc = create_tensor_descriptor(&[batch, m, k], &[m*k, k, 1], 1, alignment, input_type)?;
    let b_desc = create_tensor_descriptor(&[batch, k, n], &[k*n, n, 1], 2, alignment, input_type)?;
    let a_dot_b_desc = create_virtual_tensor_descriptor(&[batch, m, n], &[m*n, n, 1], 3, alignment, intermediate_type)?;
    let bias_desc = create_tensor_descriptor(&[1, m, n], &[m*n, n, 1], 4, alignment, input_type)?; // TODO check dim
    let a_dot_b_and_bias_desc = create_virtual_tensor_descriptor(&[batch, m, n], &[m*n, n, 1], 5, alignment, input_type)?;
    let result_desc = create_tensor_descriptor(&[batch, m, n], &[m*n, n, 1], 6, alignment, input_type)?;

    let matmul_desc = MatmulDescriptorBuilder::new()?
        .set_compute_type(&compute_type)?
        .finalize()?;
    let matmul_op_desc = OperationMatmulDescriptorBuilder::new()?
        .set_matmul_desc(&matmul_desc)?
        .set_a_descriptor(&a_desc)?
        .set_b_descriptor(&b_desc)?
        .set_c_descriptor(&a_dot_b_desc)?
        .finalize()?;    

    let add_desc = PointwiseDescriptorBuilder::new()?
        .set_mode(sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_ADD)?
        .finalize()?;
    let add_op_desc = OperationPointwiseDescriptorBuilder::new()?
        .set_pw_descriptor(&add_desc)?
        .set_x_descriptor(&a_dot_b_desc)?
        .set_b_descriptor(&bias_desc)?
        .set_y_descriptor(&a_dot_b_and_bias_desc)?
        // .set_math_prec(&intermediate_type)?
        .finalize()?;

    let relu_desc = PointwiseDescriptorBuilder::new()?
        .set_mode(sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_RELU_FWD)?
        .finalize()?;
    let relu_op_desc = OperationPointwiseDescriptorBuilder::new()?
        .set_pw_descriptor(&relu_desc)?
        .set_x_descriptor(&a_dot_b_and_bias_desc)?
        .set_y_descriptor(&result_desc)?
        .finalize()?;

    let handle = create_handle()?;

    let graph_desc = OperationGraphDescriptorBuilder::new()?
        // .set_ops(&matmul_op_desc, &[&add_op_desc])?
        // .set_ops(&matmul_op_desc, &[&add_op_desc, &relu_op_desc])?
        .set_ops_matmul(&matmul_op_desc)?
        .set_ops_pointwise(&add_op_desc)?
        .set_ops_pointwise(&relu_op_desc)?
        .set_handle(&handle)?
        .finalize()?;

    let engine_count = graph_desc.get_engine_glocal_count()?;
    dbg!(engine_count);

    // todo heuristics don't work ?
    let heuristics_desc = EngineHeuristicsDescriptorBuilder::new()?
        .set_operation_graph(&graph_desc)?
        .set_mode(&cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_A)?
        .finalize()?
        .get_results(engine_count)?;

    dbg!(&heuristics_desc.len());

    for heuristic in heuristics_desc {
        let engine = heuristic.get_engine()?;
        dbg!(engine.get_global_index()?);
        dbg!(engine.get_numerical_note()?);
    }


    // let engine_desc = EngineDescriptorBuilder::new()?
    //     .set_operation_graph(&graph_desc)?
    //     .set_global_index(0)?
    //     .finalize()?;

    // let engcfg = EngineConfigDescriptorBuilder::new()?
    //     .set_engine(&engine_desc)?
    //     .finalize()?;

    // let plan = ExecutionPlanDescriptorBuilder::new()?
    //     .set_handle(&handle)?
    //     .set_engine_config(&engcfg)?
    //     .finalize()?;
    // let workspace_size = plan.get_workspace_size()?;
    // dbg!(workspace_size);

    // // Allocate CUDA memory
    // let uids = [1, 2, 3];
    // let dev = CudaDevice::new(0)?;
    // let mut b = dev.alloc_zeros::<f32>(2)?;
    // let mut c = dev.alloc_zeros::<f32>(2)?;
    // let mut d = dev.alloc_zeros::<f32>(2)?;
    // let mut workspace = dev.alloc_zeros::<f32>(2)?;
    // dev.htod_copy_into(vec![3.0; 2], &mut b)?;
    // dev.htod_copy_into(vec![2.0; 2], &mut c)?;
    // dev.htod_copy_into(vec![0.0; 2], &mut d)?;
    // dev.htod_copy_into(vec![0.0; 2], &mut workspace)?;

    // let mut dev_ptrs = [&mut b, &mut c, &mut d];
    // let variant_pack_desc = VariantPackDescriptorBuilder::new()?
    //     .set_unique_ids(&uids)?
    //     .set_workspace(&mut workspace)?
    //     .set_data_pointers(&mut dev_ptrs)?
    //     .finalize()?;

    // // run the graph
    // backend_execute(handle, plan.descriptor, variant_pack_desc.descriptor)?;
    // let mut d_host = vec![0.0; 2];
    // dev.dtoh_sync_copy_into(&d, &mut d_host)?;
    // dbg!(&d_host);

    // // run the graph again
    // dev.htod_copy_into(vec![4.0; 2], &mut b)?;
    // backend_execute(handle, plan.descriptor, variant_pack_desc.descriptor)?;
    // dev.dtoh_sync_copy_into(&d, &mut d_host)?;
    // dbg!(d_host);

    Ok(())
}
