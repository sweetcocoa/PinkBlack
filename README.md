# PinkBlack #
----
내가 딥 러닝 개발하는데 필요한 함수 모음

1. argument parse
- example) 
```bash
python main.py --model resnet --gpu 2
```

```python
import PinkBlack, os
args = PinkBlack.io.setup(default_args={'Model':"alexnet", "gpu":"1,2"})
print(args.model) # "resnet"
print(args.gpu) # "1, 2"
print(os.environ['CUDA_VISIBLE_DEVICE']) # 2
```

2. Image process
- 얼굴 랜드마크 기준으로 이미지 Align
- 이미지 비율 유지하며 정사각 패딩, 리사이즈
- 이미지 중심점 기준 크롭

3. notebook 관련
- 이미지, 타일 수 지정으로 한 번에 visualize
