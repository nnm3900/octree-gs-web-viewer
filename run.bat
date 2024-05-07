@echo off

call conda activate octree-viewer

python convert-gs-data.py

python -m onnxruntime.tools.convert_onnx_models_to_ort ./converted/color_mlp.onnx
python -m onnxruntime.tools.convert_onnx_models_to_ort ./converted/cov_mlp.onnx
python -m onnxruntime.tools.convert_onnx_models_to_ort ./converted/opacity_mlp.onnx

echo Both scripts executed successfully.
pause