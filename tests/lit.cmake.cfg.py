import os
import lit.formats
from lit.llvm import llvm_config

print(f"mlir_obj_dir: {config.mlir_obj_dir}")
print(f"project_binary_dir: {config.project_binary_dir}")

config.name = "EINSUM_TESTS"

config.test_format = lit.formats.ShTest()

config.suffixes = ['.mlir']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.project_binary_dir, "tests")
config.project_tools_dir = os.path.join(config.project_binary_dir, "tools/einsum-opt/")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%project_source_dir", config.project_source_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.project_tools_dir]
tools = ["einsum-opt"]
llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(config.project_binary_dir, "python_packages"),
        os.path.join(config.project_binary_dir, "python_packages", "einsum_mlir"),
    ],
    append_path=True,
)
