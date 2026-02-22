import argparse
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet

def main(args):
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
            args.output, 
            opset_version=14, # TensorRT usually prefers 13 or 14 for stability
            export_params=True,
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            },
            input_names=['input'],
            output_names=['output']
        )
    print(torch_out.shape)

if __name__ == '__main__':
    """Convert pytorch model to onnx models with dynamic axes"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, default='weights/RealESRGAN_x4plus.pth', help='Input model path')
    parser.add_argument('--output', type=str, default='/output/realesrgan-x4plus.onnx', help='Output onnx path')
    parser.add_argument('--params', action='store_false', help='Use params instead of params_ema')
    args = parser.parse_args()

    main(args)
