# ComfyUI-GGUF-Next || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# pytest glue: test_ops.py needs a ComfyUI checkout (COMFYUI_PATH), test_dequant.py doesn't
import pytest

@pytest.fixture(scope="session")
def _comfy():
    import test_ops
    ops, dq = test_ops.setup()
    if ops is None:
        pytest.skip("COMFYUI_PATH not set")
    return ops, dq

@pytest.fixture
def ops(_comfy):
    return _comfy[0]

@pytest.fixture
def dq(_comfy):
    return _comfy[1]
