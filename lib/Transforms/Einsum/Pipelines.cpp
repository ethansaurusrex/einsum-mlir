#include "lib/Transforms/Einsum/Passes.h"
#include "lib/Transforms/Einsum/PTX/Passes.h"
#include "lib/Conversion/EinsumToLinalg/EinsumToLinalg.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/include/mlir/Dialect/Linalg/Passes.h"
#include "mlir/include/mlir/Dialect/Affine/Passes.h"
#include "mlir/include/mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/include/mlir/Conversion/Passes.h"
#include "mlir/include/mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"

using namespace mlir;

namespace mlir::einsum {

static void registerEinsumToLinalgPipeline() {
  PassPipelineRegistration<>(
      "einsum-to-linalg-pipeline",
      "Lower Einsum Dialect to Linalg",
      [&](OpPassManager &pm) {
        pm.addPass(createEinsumHLToLL());
        pm.addPass(createEinsumToLinalg());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
      });
}

static void registerLinalgToPTXPipeline() {
  PassPipelineRegistration<>(
      "linalg-to-ptx-pipeline",
      "Lower Linalg Generic to PTX",
      [&](OpPassManager &pm) {
	pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
	pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
	{ // limits opts scope
	  mlir::bufferization::OneShotBufferizePassOptions opts;
	  opts.bufferizeFunctionBoundaries = true;
	pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));
	}
	pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
	pm.addPass(createParallelLoopsTracker());
	pm.addPass(createParallelLoopsRemover());
	pm.addPass(createPtxCodegen());
      });
}

static void registerEinsumToPTXPipeline() {
  PassPipelineRegistration<>(
      "einsum-to-ptx-pipeline",
      "Lower Einsum dialect to PTX",			     
      [&](OpPassManager &pm) {
	// einsum-to-linalg-pipeline
        pm.addPass(createEinsumHLToLL());
        pm.addPass(createEinsumToLinalg());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
	// linalg-to-ptx-pipeline
	pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
	pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
	{ // limits opts scope
	  mlir::bufferization::OneShotBufferizePassOptions opts;
	  opts.bufferizeFunctionBoundaries = true;
	pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));
	}
	pm.nest<func::FuncOp>().addPass(mlir::createConvertLinalgToParallelLoopsPass());
	pm.addPass(createParallelLoopsTracker());
	pm.nest<func::FuncOp>().addPass(createParallelLoopsRemover());
	pm.addPass(createPtxCodegen());
      });			     
} 
    

static void registerLinalgToLLVMPipeline() {
  PassPipelineRegistration<>(
      "linalg-to-llvm-pipeline",
      "Lower Linalg and Func to LLVM IR",
      [](OpPassManager &pm) {
        auto pipeline = 
          "one-shot-bufferize{bufferize-function-boundaries},"
          "convert-linalg-to-loops,"
          "convert-scf-to-cf,"
          "convert-cf-to-llvm,"
          "convert-math-to-llvm,"
          "convert-arith-to-llvm,"
          "convert-func-to-llvm,"
          "finalize-memref-to-llvm,"
          "reconcile-unrealized-casts";

        if (failed(parsePassPipeline(pipeline, pm)))
          llvm::report_fatal_error("Invalid Linalg->LLVM pipeline");
      });
}

struct PTXPipelineOptions : public mlir::PassPipelineOptions<PTXPipelineOptions> {
  Option<std::string> chip{
      *this, "chip",
      llvm::cl::desc("PTX target chip"),
      llvm::cl::init("sm_70")};

  Option<std::string> ptxVersion{
      *this, "ptx-version",
      llvm::cl::desc("PTX feature version"),
      llvm::cl::init("+ptx80")};
};

static void registerLinalgToPTXPipelineBuiltin() {
  /*
    Below pass is based off of a PTX lowering by Stephen Diehl
    discussed and implemented:
    https://www.stephendiehl.com/posts/mlir_gpu/

    It should be ran and the output piped into
    llc -march=nvptx64 -mcpu={chip_type}
    where chip_type matches the 'chip' string passed below
  */
  PassPipelineRegistration<PTXPipelineOptions>(
      "linalg-to-ptx-pipeline-builtin",
      "Lower Linalg to GPU -> NVVM -> PTX using builtin pipeline",
      [](OpPassManager &pm, const PTXPipelineOptions &options) {
	mlir::bufferization::OneShotBufferizePassOptions bufferizeOptions;
	bufferizeOptions.functionBoundaryTypeConversion =
	  mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
	
        pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));
	
        pm.addPass(createConvertLinalgToAffineLoopsPass());

        pm.nest<func::FuncOp>().addPass(affine::createAffineLoopInvariantCodeMotionPass());
        pm.nest<func::FuncOp>().addPass(createConvertAffineForToGPUPass());

	pm.addPass(createGpuKernelOutliningPass());

        pm.addPass(createLowerAffinePass());

        pm.addPass(createGpuDecomposeMemrefsPass());
        pm.addPass(memref::createExpandStridedMetadataPass());
        pm.addPass(memref::createNormalizeMemRefsPass());

	ConvertGpuOpsToNVVMOpsOptions gpuToNVVMoptions;
	gpuToNVVMoptions.indexBitwidth = 0;
	gpuToNVVMoptions.useBarePtrCallConv = 0;	
        pm.nest<gpu::GPUModuleOp>().addPass(createConvertGpuOpsToNVVMOps(gpuToNVVMoptions));

	GpuNVVMAttachTargetOptions gpuNVVMAttachoptions;
	gpuNVVMAttachoptions.chip = options.chip;
	gpuNVVMAttachoptions.features = options.ptxVersion;
	gpuNVVMAttachoptions.optLevel = 3;
	
        pm.addPass(createGpuNVVMAttachTarget(gpuNVVMAttachoptions));

        pm.addPass(createConvertNVVMToLLVMPass());

        pm.addPass(createReconcileUnrealizedCastsPass());

	GpuToLLVMConversionPassOptions gpuToLLVMoptions;
	gpuToLLVMoptions.hostBarePtrCallConv = true;
	gpuToLLVMoptions.kernelBarePtrCallConv = true;
	
        pm.addPass(createGpuToLLVMConversionPass(gpuToLLVMoptions));
      });
}


void registerEinsumPipelines() {
  registerEinsumToLinalgPipeline();
  registerLinalgToPTXPipeline();
  registerEinsumToPTXPipeline();
  registerLinalgToLLVMPipeline();
  registerLinalgToPTXPipelineBuiltin();
}

} // namespace einsum::mlir
