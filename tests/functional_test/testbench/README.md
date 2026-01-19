# Functional tests

## Preparation

To avoid conflict tests on different operators, we use a private `test_env.sh` file in your local environment to control which op to test. Create one based on `test_env_sample.sh` file.

### Configuration variables in `test_env.sh`

The `test_env.sh` file contains several environment variables that control the behavior of the functional tests:

- **TEST_OPS_LIST**:  
  Specifies which operators to test.  
  *Type*: comma-separated list of operator names (e.g., `MatMul,Conv2D`).  
  *Valid values*: Any subset of supported operator names.  
  *Purpose*: Limits the test run to only the specified operators.

- **TEST_CASE_IDXES**:  
  Specifies which test cases to run for the selected operators.  
  *Type*: comma-separated list of integer indices (e.g., `0,1,2`).  
  *Valid values*: Non-negative integers corresponding to available test cases.  
  *Purpose*: Allows running a subset of test cases for each operator.

- **TP_COMM_OVERLAP**:  
  Enables or disables Tensor Parallel communication overlap optimization.  
  *Type*: Boolean (`true` or `false`, or `1` or `0`).  
  *Valid values*: `true`/`1` to enable, `false`/`0` to disable.  
  *Purpose*: When enabled, overlaps communication and computation in tensor parallel tests, which may improve performance.  
  *When to enable*: Enable if you want to benchmark or debug communication overlap in tensor parallel scenarios. Disable for baseline or non-tensor-parallel tests.

### Relationship to `AutoTuner/testbench/profile/configs/override_tf_config.json`

The `test_env.sh` file sets environment variables that control which tests are run and how they are executed. The JSON config file at `AutoTuner/testbench/profile/configs/override_tf_config.json` provides additional configuration for the testbench, such as TensorFlow-specific overrides.  

- Use `test_env.sh` to select operators, test cases, and enable/disable features for the test run.
- Use `override_tf_config.json` to customize the underlying framework or testbench behavior.

Both files may affect the test outcomes, but `test_env.sh` is focused on test selection and feature toggles, while the JSON config is for framework-level settings.