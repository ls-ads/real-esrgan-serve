import torch
import torch.onnx
import onnx
import os
from onnxconverter_common import float16
from basicsr.archs.rrdbnet_arch import RRDBNet

def main(args):
    # Determine output filename based on precision
    suffix = '_fp16' if args.half else '_fp32'
    output_path = f"/output/realesrgan-x4plus{suffix}.onnx"

    # An instance of the model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    if args.params:
        keyname = 'params'
    else:
        keyname = 'params_ema'
    model.load_state_dict(torch.load(args.input)[keyname])
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()

    # Dynamic input: batch size, channels, height, width
    # Real-ESRGAN requires static channel dimension (3), but spatial and batch can be dynamic
    x = torch.rand(1, 3, 64, 64)
    
    # Export the model
    with torch.no_grad():
        torch_out = torch.onnx._export(
            model, 
            x, 
            output_path, 
            opset_version=14, # TensorRT usually prefers 13 or 14 for stability
            export_params=True,
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            },
            input_names=['input'],
            output_names=['output']
        )
    print(f"Model exported to {output_path}")

    if args.half:
        print("Converting to half precision (FP16)...")
        model_fp32 = onnx.load(output_path)
        model_fp16 = float16.convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, output_path)
        print("Conversion successful.")

if __name__ == '__main__':
    """Convert pytorch model to onnx models with dynamic axes"""
    import argparse
    parser = argparse.ArgumentParser()
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument(
        '--input', type=str, default='weights/RealESRGAN_x4plus.pth', help='Input model path')
    parser.add_argument('--params', action='store_false', help='Use params instead of params_ema')
    parser.add_argument('--half', type=str2bool, default=True, help='Export with half precision (FP16)')
    args = parser.parse_args()

    main(args)
