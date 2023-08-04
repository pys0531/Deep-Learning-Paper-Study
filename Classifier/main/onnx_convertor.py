import torch
import torch.onnx

from config import cfg
from model import get_network

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_network()
    model.to(device)

    x = torch.randn(1, 3, cfg.input_shape[0], cfg.input_shape[1], requires_grad=True).to(device)
    torch.onnx.export(model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  f"{cfg.network}_{cfg.network_type}.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

    
            
if __name__ == "__main__":
    main()