build:
	cargo build --features=cuda-12060,cudnn,f16,nvtx

test:
	cargo test --features=cuda-12060,cudnn,f16,nvtx

example-nvtx:
	cargo run --features=cuda-12060,cudnn,f16,nvtx --release --example nvtx-range

test_half:
	cargo test --features=cuda-12060,cudnn,f16,nvtx driver::safe::launch::tests::test_launch_with_half

test-conv:
	cargo test --features=cuda-12060,cudnn,f16,nvtx cudnn::safe::tests::test_conv2d_pick_algorithms

example:
	RUST_BACKTRACE=1 \
	CUDNN_LOGLEVEL_DBG=3 \
	CUDNN_LOGDEST_DBG=stdout \
	cargo run --features=cuda-12060,cudnn,nvtx --release --example cudnn-graph


examplen:
	cargo run --features=cuda-12060,cudnn,nvtx --release --example cudnn-graph
