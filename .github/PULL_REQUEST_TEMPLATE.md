## Description

Please include a summary of the changes and the related issue (if applicable).
Fixes # (issue)

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring (no functional changes)

## Component

- [ ] Python engine (`deepcgh_engine/engine.py`)
- [ ] C++ engine (`DeepCGHEngine/src/`)
- [ ] Multi-wavelength module (`deepcgh_engine/multi_wavelength.py`)
- [ ] SLM driver (`deepcgh_engine/slm_driver.py`)
- [ ] Realtime module (`deepcgh_engine/realtime.py`)
- [ ] Build system (CMake / setup.py)
- [ ] Tests
- [ ] Other: ___

## Testing

Please describe the tests that you ran to verify your changes.

- [ ] `test_engine.py` passes
- [ ] `test_integration.py` passes
- [ ] C++ build succeeds (`cmake --build build`)
- [ ] New test(s) added for the change

### Test Environment

- **OS**:
- **Python version**:
- **ONNX Runtime version**:

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
