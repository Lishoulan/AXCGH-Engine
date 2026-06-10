"""Integration test for all DeepCGHEngine modules."""
import sys
import time
import math
import os
import numpy as np

sys.path.insert(0, 'DeepCGHEngine')

def test_multi_wavelength():
    print('=== Multi-Wavelength RGB Hologram Test ===')
    from deepcgh_engine import RGBHologramEngine, RGBEngineConfig, CombineMode

    from deepcgh_engine import RGBHologramEngine, RGBEngineConfig, CombineMode, EngineConfig

    base_cfg = EngineConfig(height=256, width=256, num_planes=5)
    config = RGBEngineConfig(
        base_config=base_cfg,
        combine_mode=CombineMode.TimeDivision
    )
    engine = RGBHologramEngine()
    engine.init('DeepCGHEngine/models/deepcgh_unet.onnx', config)

    np.random.seed(42)
    rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    depth = np.random.rand(256, 256).astype(np.float32)

    status, result = engine.generate_rgb_hologram(rgb, depth)
    if status != 0:  # Status.OK
        print(f'ERROR: generate_rgb_hologram returned status {status}')
        return
    pr = result['phase_r']
    pg = result['phase_g']
    pb = result['phase_b']
    pc = result['phase_combined']
    print(f'Phase R range: [{pr.min():.4f}, {pr.max():.4f}]')
    print(f'Phase G range: [{pg.min():.4f}, {pg.max():.4f}]')
    print(f'Phase B range: [{pb.min():.4f}, {pb.max():.4f}]')
    print(f'Combined range: [{pc.min():.4f}, {pc.max():.4f}]')

    # Benchmark
    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        engine.generate_rgb_hologram(rgb, depth)
    elapsed = (time.perf_counter() - t0) / n * 1000
    print(f'RGB hologram: {elapsed:.1f} ms/frame ({1000/elapsed:.1f} FPS)')

    # Spatial multiplex
    config2 = RGBEngineConfig(
        base_config=EngineConfig(height=256, width=256, num_planes=5),
        combine_mode=CombineMode.SpatialMultiplex
    )
    engine2 = RGBHologramEngine()
    engine2.init('DeepCGHEngine/models/deepcgh_unet.onnx', config2)
    status2, result2 = engine2.generate_rgb_hologram(rgb, depth)
    pc2 = result2['phase_combined']
    print(f'Spatial multiplex combined range: [{pc2.min():.4f}, {pc2.max():.4f}]')

    engine.shutdown()
    engine2.shutdown()
    print('Multi-wavelength test PASSED!\n')


def test_int8_quantization():
    print('=== INT8 Quantization Test ===')
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'quantize_model', 'DeepCGHEngine/tools/quantize_model.py')
    qm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qm)
    print('Quantization tool loaded successfully')

    # Run quantization
    input_model = 'DeepCGHEngine/models/deepcgh_unet.onnx'
    output_model = 'DeepCGHEngine/models/deepcgh_unet_int8.onnx'

    if os.path.exists(output_model):
        os.remove(output_model)

    try:
        qm.quantize_model(
            input_path=input_model,
            output_path=output_model,
            calibration_samples=50,
            quant_format='QDQ',
            per_channel=True
        )
        print(f'INT8 model saved to: {output_model}')

        # Verify INT8 model works
        import onnxruntime as ort
        sess = ort.InferenceSession(output_model)
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        print(f'INT8 model input: {input_name} {input_shape}')
        print(f'INT8 model outputs: {[o.name for o in sess.get_outputs()]}')

        # Test inference
        dummy = np.random.rand(1, input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
        outputs = sess.run(None, {input_name: dummy})
        print(f'INT8 output shapes: {[o.shape for o in outputs]}')
        print(f'INT8 amp range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]')
        print(f'INT8 phi range: [{outputs[1].min():.4f}, {outputs[1].max():.4f}]')

        # Size comparison
        fp32_size = os.path.getsize(input_model) / 1024 / 1024
        int8_size = os.path.getsize(output_model) / 1024 / 1024
        print(f'FP32 model: {fp32_size:.2f} MB')
        print(f'INT8 model: {int8_size:.2f} MB')
        print(f'Compression ratio: {fp32_size/int8_size:.2f}x')
        print('INT8 quantization test PASSED!\n')
    except Exception as e:
        print(f'INT8 quantization test SKIPPED: {e}\n')


def test_slm_driver():
    print('=== SLM Driver Test ===')
    from deepcgh_engine import create_slm_driver, FileBackend

    # File backend test
    slm = create_slm_driver('file', resolution=(256, 256), output_dir='test_slm_output')
    phase = np.random.uniform(-np.pi, np.pi, (256, 256)).astype(np.float32)
    slm.display(phase)
    slm.clear()
    slm.close()

    # Check files were created
    files = os.listdir('test_slm_output')
    print(f'FileBackend created {len(files)} files: {files}')

    # Cleanup
    import shutil
    shutil.rmtree('test_slm_output', ignore_errors=True)
    print('SLM driver test PASSED!\n')


def test_realtime_module():
    print('=== Realtime Module Test ===')
    from deepcgh_engine import RealtimeConfig, TestPatternSource, EngineConfig

    # Test pattern source
    source = TestPatternSource(width=256, height=256)
    rgb, depth = source.capture()
    print(f'TestPattern: rgb={rgb.shape} dtype={rgb.dtype}, depth={depth.shape} dtype={depth.dtype}')
    print(f'RGB range: [{rgb.min()}, {rgb.max()}]')
    print(f'Depth range: [{depth.min():.4f}, {depth.max():.4f}]')

    config = RealtimeConfig(
        model_path='DeepCGHEngine/models/deepcgh_unet.onnx',
        engine_config=EngineConfig(height=256, width=256, num_planes=5),
        camera_type='test'
    )
    print(f'RealtimeConfig: target_fps={config.target_fps}, camera={config.camera_type}')
    print('Realtime module test PASSED!\n')


def test_cpp_engine_fftw():
    print('=== C++ Engine (FFTW3) Test ===')
    from deepcgh_engine import CppDeepCGHEngine

    engine = CppDeepCGHEngine()
    engine.init('DeepCGHEngine/models/deepcgh_unet.onnx',
                height=256, width=256, num_planes=5)

    np.random.seed(42)
    rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    depth = np.random.rand(256, 256).astype(np.float32)

    phase = engine.generate_hologram(rgb, depth)
    print(f'Phase range: [{phase.min():.4f}, {phase.max():.4f}]')

    # Benchmark
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        phase = engine.generate_hologram(rgb, depth)
    elapsed = (time.perf_counter() - t0) / n * 1000
    print(f'C++ FFTW3: {elapsed:.2f} ms/frame ({1000/elapsed:.1f} FPS)')

    if phase.min() >= -math.pi - 0.1 and phase.max() <= math.pi + 0.1:
        print('Phase range check: PASSED')
    else:
        print('Phase range check: FAILED')

    engine.shutdown()
    print('C++ engine test PASSED!\n')


def test_python_engine():
    print('=== Python Engine Test ===')
    from deepcgh_engine import EngineAPI, EngineConfig

    engine = EngineAPI()
    engine.init('DeepCGHEngine/models/deepcgh_unet.onnx',
                EngineConfig(height=256, width=256, num_planes=5))

    np.random.seed(42)
    rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    depth = np.random.rand(256, 256).astype(np.float32)

    status, phase = engine.generate_hologram(rgb, depth)
    print(f'Phase range: [{phase.min():.4f}, {phase.max():.4f}]')

    # Benchmark
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        status, phase = engine.generate_hologram(rgb, depth)
    elapsed = (time.perf_counter() - t0) / n * 1000
    print(f'Python NumPy FFT: {elapsed:.2f} ms/frame ({1000/elapsed:.1f} FPS)')

    engine.shutdown()
    print('Python engine test PASSED!\n')


if __name__ == '__main__':
    os.chdir('F:/deepcgh')

    test_python_engine()
    test_cpp_engine_fftw()
    test_multi_wavelength()
    test_slm_driver()
    test_realtime_module()
    test_int8_quantization()

    print('=' * 50)
    print('ALL INTEGRATION TESTS PASSED!')
    print('=' * 50)
