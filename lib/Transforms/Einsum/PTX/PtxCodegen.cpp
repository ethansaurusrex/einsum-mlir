#include "lib/Transforms/Einsum/PTX/PtxCodegen.h"
#include "lib/Transforms/Einsum/PTX/ParallelLoopsTracker.h"

namespace mlir {
namespace einsum {

class PTXCodeGenerator {
public:
  PTXCodeGenerator() : regCounter(0), labelCounter(0), paramCounter(0) {}
  
  std::string generate(ModuleOp module) {
    std::stringstream ptx;
    
    // PTX Header
    ptx << "//\n";
    ptx << ".version 7.0\n";
    ptx << ".target sm_75\n";
    ptx << ".address_size 64\n\n";
    
    module.walk([&](func::FuncOp funcOp) {
      generateFunction(funcOp, ptx);
      generateLauncher(funcOp, ptx);
    });
    
    return ptx.str();
  }

private:
  unsigned regCounter;
  unsigned labelCounter;
  unsigned paramCounter;
  DenseMap<Value, std::string> valueMap;
  
  std::string getNewReg(Type type) {
    std::string prefix = getPTXTypePrefix(type);
    return "%" + prefix + std::to_string(regCounter++);
  }
  
  std::string getNewLabel() {
    return "BB" + std::to_string(labelCounter++);
  }
  
  std::string getPTXTypePrefix(Type type) {
    if (type.isInteger(1)) return "p";      // predicate
    if (type.isInteger(32)) return "r";     // 32-bit register
    if (type.isInteger(64)) return "rd";    // 64-bit register
    if (type.isF32()) return "f";           // float
    if (type.isF64()) return "fd";          // double
    if (isa<mlir::MemRefType>(type)) return "rd";
    return "r"; // default
  }
  
  std::string getPTXType(Type type) {
    if (type.isInteger(1)) return "pred";
    if (type.isInteger(8)) return "u8";
    if (type.isInteger(16)) return "u16";
    if (type.isInteger(32)) return "u32";
    if (type.isInteger(64)) return "u64";
    if (type.isF32()) return "f32";
    if (type.isF64()) return "f64";
    if (isa<mlir::MemRefType>(type)) return "u64";
    return "u32"; // addresses? 
  }
  
  void generateFunction(func::FuncOp funcOp, std::stringstream &ptx) {
    // Reset counters for each function
    regCounter = 0;
    labelCounter = 0;
    paramCounter = 0;
    valueMap.clear();
    
    ptx << "// Function: " << funcOp.getName().str() << "_kernel" << "\n";

    // always a kernel and not a GPU function for simplicity
    ptx << ".visible .entry ";
    
    ptx << funcOp.getName().str() << "(\n";

    // parameters
    auto funcType = funcOp.getFunctionType();
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      Type paramType = funcType.getInput(i);
      std::string paramName = ".param .u64 param_" + std::to_string(i);
      ptx << "    " << paramName;
      if (i < funcType.getNumInputs() - 1) ptx << ",";
      ptx << "\n";
    }
    ptx << ")\n{\n";

    // register declarations
    generateRegisterDeclarations(funcOp, ptx);
    ptx << "\n";
    
    // load function arguments to registers
    Block &entryBlock = funcOp.front();
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      BlockArgument arg = entryBlock.getArgument(i);
      llvm::errs() << "arg: \n";
      arg.dump();
      llvm::errs() << arg.getType() << "\n";
      std::string reg = getNewReg(arg.getType());
      valueMap[arg] = reg;
      ptx << "        ld.param." << getPTXType(arg.getType()) 
          << " " << reg << ", [param_" << i << "];\n";
    }
    
    ptx << "\n";
    
    // body
    generateBlock(entryBlock, ptx);
    
    ptx << "}\n\n";
  }
  
  void generateRegisterDeclarations(func::FuncOp funcOp, std::stringstream &ptx) {
    // unsigned numPredicates = 0;
    // unsigned numInt32 = 0;
    // unsigned numInt64 = 0;
    // unsigned numFloat32 = 0;
    // unsigned numFloat64 = 0;
    
    // funcOp.walk([&](Operation *op) {
    //   for (Value result : op->getResults()) {
    //     Type type = result.getType();
    //     if (type.isInteger(1)) numPredicates++;
    //     else if (type.isInteger(32)) numInt32++;
    //     else if (type.isInteger(64)) numInt64++;
    //     else if (type.isF32()) numFloat32++;
    //     else if (type.isF64()) numFloat64++;
    //   }
    // });
    
    // // Add some buffer
    // numPredicates += 10;
    // numInt32 += 20;
    // numInt64 += 10;
    
    unsigned numPredicates = 30;
    unsigned numInt32 = 30;
    unsigned numInt64 = 30;
    unsigned numFloat32 = 30;
    unsigned numFloat64 = 30;

    if (numPredicates > 0)
      ptx << "        .reg .pred %p<" << numPredicates << ">;\n";
    if (numInt32 > 0)
      ptx << "        .reg .u32 %r<" << numInt32 << ">;\n";
    if (numInt64 > 0)
      ptx << "        .reg .u64 %rd<" << numInt64 << ">;\n";
    if (numFloat32 > 0)
      ptx << "        .reg .f32 %f<" << numFloat32 << ">;\n";
    if (numFloat64 > 0)
      ptx << "        .reg .f64 %fd<" << numFloat64 << ">;\n";
  }
  
  void generateBlock(Block &block, std::stringstream &ptx) {

    for (auto it = block.begin(); it != block.end(); ++it) {
        mlir::Operation &currentOp = *it;
        
        if (auto currentPoison = llvm::dyn_cast<ub::PoisonOp>(currentOp)) {
            auto nextIt = std::next(it);
            if (nextIt != block.end()) {
                mlir::Operation &nextOp = *nextIt;
                
                if (auto nextPoison = llvm::dyn_cast<ub::PoisonOp>(nextOp)) {
                    generateId(currentPoison, nextPoison, ptx);
                }
            }
        } else {
          generateOperation(&currentOp, ptx);
        }
    }

  }
  
  void generateOperation(Operation *op, std::stringstream &ptx) {
    llvm::TypeSwitch<Operation *>(op)
      .Case<arith::ConstantOp>([&](auto constOp) {
        generateConstant(constOp, ptx);
      })
      .Case<arith::AddIOp>([&](auto addOp) {
        generateBinaryOp(addOp, "add", ptx);
      })
      .Case<arith::SubIOp>([&](auto subOp) {
        generateBinaryOp(subOp, "sub", ptx);
      })
      .Case<arith::MulIOp>([&](auto mulOp) {
        generateBinaryOp(mulOp, "mul.lo", ptx);
      })
      .Case<arith::DivSIOp>([&](auto divOp) {
        generateBinaryOp(divOp, "div", ptx);
      })
      .Case<arith::CmpIOp>([&](auto cmpOp) {
        generateCmpOp(cmpOp, ptx);
      })
      .Case<arith::AddFOp>([&](auto addOp) {
        generateBinaryOp(addOp, "add", ptx);
      })
      .Case<arith::MulFOp>([&](auto mulOp) {
        generateBinaryOp(mulOp, "mul", ptx);
      })
      .Case<scf::ForOp>([&](auto forOp) {
        generateForLoop(forOp, ptx);
      })
      .Case<scf::IfOp>([&](auto ifOp) {
        generateIf(ifOp, ptx);
      })
      .Case<scf::WhileOp>([&](auto whileOp) {
        generateWhile(whileOp, ptx);
      })
      .Case<scf::YieldOp>([&](auto yieldOp) {
        // Handle in parent operation
      })
      .Case<memref::LoadOp>([&](auto loadOp) {
        generateLoad(loadOp, ptx);
      })
      .Case<memref::StoreOp>([&](auto storeOp) {
        generateStore(storeOp, ptx);
      })
      .Case<func::ReturnOp>([&](auto returnOp) {
        ptx << "        ret;\n";
      })
      .Case<ub::PoisonOp>([&](auto poisonOp) {
        // do nothing here
      })
      .Default([&](Operation *op) {
        ptx << "    // Unsupported operation: " << op->getName().getStringRef().str() << "\n";
      });
  }

  void generateId(ub::PoisonOp& first, ub::PoisonOp& second, std::stringstream &ptx) {

    if (Attribute gpuAttr1 = first->getAttr("GPU")) {
      if (Attribute gpuAttr2 = second->getAttr("GPU")) {
        // threadIdx.x
        std::string reg1 = getNewReg(first.getType());
        ptx << "        mov.u32" << " " 
            << reg1 << ", " << "%tid.x" << ";\n";
        // blockIdx.x
        std::string reg2 = getNewReg(second.getType());
        ptx << "        mov.u32" << " " 
            << reg2 << ", " << "%ctaid.x" << ";\n";
        // blockDim.x
        std::string reg3 = getNewReg(second.getType());
        ptx << "        mov.u32" << " " 
            << reg3 << ", " << "%ntid.x" << ";\n";

        // globalId = blockIdx.x * blockDim.x + threadIdx.x
        std::string reg4 = getNewReg(second.getType());
        ptx << "        mul.lo.u32" << " " 
            << reg4 << ", " << reg2 << ", " << reg3 <<  ";\n";
        std::string reg5 = getNewReg(second.getType());
        ptx << "        add.u32" << " " 
            << reg5 << ", " << reg4 << ", " << reg1 <<  ";\n";
        
        // row = globalId / width
        std::string reg6 = getNewReg(first.getType());
        valueMap[first.getResult()] = reg6;
        ptx << "        div.u32" << " " 
            << reg6 << ", " << reg5 << ", " << "" <<  ";\n";

        std::string reg7 = getNewReg(second.getType());
        valueMap[second.getResult()] = reg7;
        ptx << "        rem.u32" << " " 
            << reg7 << ", " << reg5 << ", " << "" << ";\n";
      }
    }
  }
  
  void generateConstant(arith::ConstantOp constOp, std::stringstream &ptx) {
    std::string reg = getNewReg(constOp.getType());
    valueMap[constOp.getResult()] = reg;
    
    auto attr = constOp.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      ptx << "        mov." << getPTXType(constOp.getType()) << " " 
          << reg << ", " << intAttr.getInt() << ";\n";
    } else if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
      ptx << "        mov." << getPTXType(constOp.getType()) << " " 
          << reg << ", " << floatAttr.getValueAsDouble() << ";\n";
    }
  }
  
  template<typename OpTy>
  void generateBinaryOp(OpTy op, const std::string &ptxOp, std::stringstream &ptx) {
    std::string lhs = valueMap[op.getLhs()];
    std::string rhs = valueMap[op.getRhs()];
    std::string result = getNewReg(op.getType());
    valueMap[op.getResult()] = result;
    
    ptx << "        " << ptxOp << "." << getPTXType(op.getType()) << " "
        << result << ", " << lhs << ", " << rhs << ";\n";
  }
  
  void generateCmpOp(arith::CmpIOp cmpOp, std::stringstream &ptx) {
    std::string lhs = valueMap[cmpOp.getLhs()];
    std::string rhs = valueMap[cmpOp.getRhs()];
    std::string result = getNewReg(cmpOp.getType());
    valueMap[cmpOp.getResult()] = result;
    
    std::string predicate;
    switch (cmpOp.getPredicate()) {
      case arith::CmpIPredicate::eq: predicate = "eq"; break;
      case arith::CmpIPredicate::ne: predicate = "ne"; break;
      case arith::CmpIPredicate::slt: predicate = "lt"; break;
      case arith::CmpIPredicate::sle: predicate = "le"; break;
      case arith::CmpIPredicate::sgt: predicate = "gt"; break;
      case arith::CmpIPredicate::sge: predicate = "ge"; break;
      default: predicate = "eq";
    }
    
    ptx << "        setp." << predicate << "." << getPTXType(cmpOp.getLhs().getType())
        << " " << result << ", " << lhs << ", " << rhs << ";\n";
  }
  
  void generateForLoop(scf::ForOp forOp, std::stringstream &ptx) {
    std::string loopHeader = getNewLabel();
    std::string loopBody = getNewLabel();
    std::string loopExit = getNewLabel();
    
    // Initialize induction variable
    std::string iv = getNewReg(forOp.getInductionVar().getType());
    valueMap[forOp.getInductionVar()] = iv;
    std::string lb = valueMap[forOp.getLowerBound()];
    std::string ub = valueMap[forOp.getUpperBound()];
    std::string step = valueMap[forOp.getStep()];
    
    ptx << "        mov." << getPTXType(forOp.getInductionVar().getType()) 
        << " " << iv << ", " << lb << ";\n";
    
    // Loop header
    ptx << "    " << loopHeader << ":\n";
    
    // Check condition
    std::string pred = getNewReg(IntegerType::get(forOp.getContext(), 1));
    ptx << "        setp.lt." << getPTXType(forOp.getInductionVar().getType())
        << " " << pred << ", " << iv << ", " << ub << ";\n";
    ptx << "        @!" << pred << " bra " << loopExit << ";\n";
    
    // Loop body
    ptx << "    " << loopBody << ":\n";
    Block &body = forOp.getRegion().front();
    for (Operation &op : body.without_terminator()) {
      generateOperation(&op, ptx);
    }
    
    // Increment
    std::string newIv = getNewReg(forOp.getInductionVar().getType());
    ptx << "        add." << getPTXType(forOp.getInductionVar().getType())
        << " " << newIv << ", " << iv << ", " << step << ";\n";
    ptx << "        mov." << getPTXType(forOp.getInductionVar().getType())
        << " " << iv << ", " << newIv << ";\n";
    
    ptx << "        bra " << loopHeader << ";\n";
    ptx << "    " << loopExit << ":\n";
  }
  
  void generateIf(scf::IfOp ifOp, std::stringstream &ptx) {
    std::string thenLabel = getNewLabel();
    std::string elseLabel = getNewLabel();
    std::string exitLabel = getNewLabel();
    
    std::string condition = valueMap[ifOp.getCondition()];
    
    // Branch to then or else
    if (ifOp.elseBlock()) {
      ptx << "        @" << condition << " bra " << thenLabel << ";\n";
      ptx << "        bra " << elseLabel << ";\n";
    } else {
      ptx << "        @" << condition << " bra " << thenLabel << ";\n";
      ptx << "        bra " << exitLabel << ";\n";
    }
    
    // Then block
    ptx << "    " << thenLabel << ":\n";
    for (Operation &op : ifOp.thenBlock()->without_terminator()) {
      generateOperation(&op, ptx);
    }
    ptx << "        bra " << exitLabel << ";\n";
    
    // Else block
    if (ifOp.elseBlock()) {
      ptx << "    " << elseLabel << ":\n";
      for (Operation &op : ifOp.elseBlock()->without_terminator()) {
        generateOperation(&op, ptx);
      }
    }
    
    ptx << "    " << exitLabel << ":\n";
  }
  
  void generateWhile(scf::WhileOp whileOp, std::stringstream &ptx) {
    std::string condLabel = getNewLabel();
    std::string bodyLabel = getNewLabel();
    std::string exitLabel = getNewLabel();
    
    ptx << condLabel << ":\n";
    
    // Generate condition block
    Block &condBlock = whileOp.getBefore().front();
    for (Operation &op : condBlock.without_terminator()) {
      generateOperation(&op, ptx);
    }
    
    // Get condition from condition op
    auto condOp = cast<scf::ConditionOp>(condBlock.getTerminator());
    std::string cond = valueMap[condOp.getCondition()];
    
    ptx << "    @" << cond << " bra " << bodyLabel << ";\n";
    ptx << "    bra " << exitLabel << ";\n";
    
    // Generate body
    ptx << bodyLabel << ":\n";
    Block &bodyBlock = whileOp.getAfter().front();
    for (Operation &op : bodyBlock.without_terminator()) {
      generateOperation(&op, ptx);
    }
    ptx << "    bra " << condLabel << ";\n";
    
    ptx << exitLabel << ":\n";
  }
  
  void generateLoad(memref::LoadOp loadOp, std::stringstream &ptx) {
    std::string result = getNewReg(loadOp.getType());
    valueMap[loadOp.getResult()] = result;
    
    std::string memref = valueMap[loadOp.getMemRef()];

    for (auto idx : llvm::reverse(loadOp.getIndices())) {
      llvm::errs() << "Index: ";
      idx.dump();
    }
    llvm::errs() << "\n\n";

    ptx << "        ld.global." << getPTXType(loadOp.getType()) << " "
        << result << ", [" << memref << "];\n";
  }
  
  void generateStore(memref::StoreOp storeOp, std::stringstream &ptx) {
    std::string value = valueMap[storeOp.getValue()];
    std::string memref = valueMap[storeOp.getMemRef()];
    
    ptx << "        st.global." << getPTXType(storeOp.getValue().getType()) << " "
        << "[" << memref << "], " << value << ";\n";
  }

  void generateLauncher(func::FuncOp funcOp, std::stringstream &ptx) {
    ptx << "void ";
    ptx << funcOp.getName().str() << "_launcher(\n";

    auto funcType = funcOp.getFunctionType();
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      std::string paramName = "float *param_" + std::to_string(i);
      ptx << "    " << paramName;
      if (i < funcType.getNumInputs() - 1) ptx << ",";
      ptx << "\n";
    }
    ptx << ")\n{\n";

    int blocks = 0;
    int threads = 0;

    ptx << "    int threads = " << threads << ";\n";
    ptx << "    int blocks = " << blocks << ";\n";

    ptx << "    " << funcOp.getName().str() << "_kernel<<<blocks, threads>>>\n";
    ptx << "    (\n";
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      std::string paramName = "param_" + std::to_string(i);
      ptx << "        " << paramName;
      if (i < funcType.getNumInputs() - 1) ptx << ",";
      ptx << "\n";
    }
    ptx << "    );\n";

    ptx << "}\n\n";
  }
};

#define GEN_PASS_DEF_PTXCODEGEN
#include "lib/Transforms/Einsum/PTX/Passes.h.inc"

struct PtxCodegen : impl::PtxCodegenBase<PtxCodegen> {
  using PtxCodegenBase::PtxCodegenBase;

  void yolo() {
    auto& parallelLoopsTrackerAnalysis = getAnalysis<ParallelLoopsTrackerAnalysis>();
    auto totalCount = parallelLoopsTrackerAnalysis.getTotalCount();
    auto functionCounts = parallelLoopsTrackerAnalysis.getFunctionCounts();
    // Print summary
    llvm::outs() << "\n┌─────────────────────────────────────────────┐\n";
    llvm::outs() << "│  SCF Parallel Loop Count Summary           │\n";
    llvm::outs() << "├─────────────────────────────────────────────┤\n";
    llvm::outs() << "│  Total parallel loops: " << totalCount << "\n";
    llvm::outs() << "│  Functions with parallel loops: " 
                 << functionCounts.size() << "\n";
    llvm::outs() << "└─────────────────────────────────────────────┘\n\n";
    
    if (!functionCounts.empty()) {
      llvm::outs() << "Per-function breakdown:\n";
      for (auto &entry : functionCounts) {
        llvm::outs() << "  " << entry.first << ":\n"; 
        // entry.second.print(llvm::outs());
        printParallelLoopInfo(entry.second, llvm::outs());
        llvm::outs() << "\n";
      }
      llvm::outs() << "\n";
    }
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    PTXCodeGenerator generator;
    std::string ptxCode = generator.generate(module);
    
    llvm::outs() << "═══════════════════════════════════════════════════════\n";
    llvm::outs() << ptxCode;
    llvm::outs() << "═══════════════════════════════════════════════════════\n\n";

    yolo();
  }
};

} // namespace einsum
} // namespace mlir
