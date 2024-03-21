/* automatically generated by rust-bindgen 0.69.4 */

pub const CUDA_VERSION: u32 = 12020;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
}
extern "C" {
    pub fn nvrtcGetErrorString(result: nvrtcResult) -> *const ::core::ffi::c_char;
}
extern "C" {
    pub fn nvrtcVersion(
        major: *mut ::core::ffi::c_int,
        minor: *mut ::core::ffi::c_int,
    ) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetNumSupportedArchs(numArchs: *mut ::core::ffi::c_int) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetSupportedArchs(supportedArchs: *mut ::core::ffi::c_int) -> nvrtcResult;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _nvrtcProgram {
    _unused: [u8; 0],
}
pub type nvrtcProgram = *mut _nvrtcProgram;
extern "C" {
    pub fn nvrtcCreateProgram(
        prog: *mut nvrtcProgram,
        src: *const ::core::ffi::c_char,
        name: *const ::core::ffi::c_char,
        numHeaders: ::core::ffi::c_int,
        headers: *const *const ::core::ffi::c_char,
        includeNames: *const *const ::core::ffi::c_char,
    ) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcCompileProgram(
        prog: nvrtcProgram,
        numOptions: ::core::ffi::c_int,
        options: *const *const ::core::ffi::c_char,
    ) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetPTXSize(prog: nvrtcProgram, ptxSizeRet: *mut usize) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut ::core::ffi::c_char) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetCUBINSize(prog: nvrtcProgram, cubinSizeRet: *mut usize) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut ::core::ffi::c_char) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetNVVMSize(prog: nvrtcProgram, nvvmSizeRet: *mut usize) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetNVVM(prog: nvrtcProgram, nvvm: *mut ::core::ffi::c_char) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetLTOIRSize(prog: nvrtcProgram, LTOIRSizeRet: *mut usize) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetLTOIR(prog: nvrtcProgram, LTOIR: *mut ::core::ffi::c_char) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetOptiXIRSize(prog: nvrtcProgram, optixirSizeRet: *mut usize) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetOptiXIR(prog: nvrtcProgram, optixir: *mut ::core::ffi::c_char) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetProgramLogSize(prog: nvrtcProgram, logSizeRet: *mut usize) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut ::core::ffi::c_char) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcAddNameExpression(
        prog: nvrtcProgram,
        name_expression: *const ::core::ffi::c_char,
    ) -> nvrtcResult;
}
extern "C" {
    pub fn nvrtcGetLoweredName(
        prog: nvrtcProgram,
        name_expression: *const ::core::ffi::c_char,
        lowered_name: *mut *const ::core::ffi::c_char,
    ) -> nvrtcResult;
}
