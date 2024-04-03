import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import argparse

arg_parser = argparse.ArgumentParser(description='Read ONNX model')
arg_parser.add_argument('-m', '--model', type=str, help='Path to ONNX model')
args = arg_parser.parse_args()


session = onnxruntime.InferenceSession(
    args.model,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

print("Inputs name:", [input.name for input in session.get_inputs()])
print("Outputs name:", [output.name for output in session.get_outputs()])


print("Inputs shape:", [input.shape for input in session.get_inputs()])
print("Outputs shape:", [output.shape for output in session.get_outputs()])