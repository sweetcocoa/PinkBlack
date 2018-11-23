# PinkBlack #
----
> 개발(주로 딥러닝)을 위한 여러 가지 잡 모듈

1. 자동 로깅 (Autolog)
2. 에러 발생할 때 자동 hooking (Auto-raise pdb)
3. 명령줄 분해 (Argument Parsing)
    ```bash
    python main.py --gpu 2 --batch_size 64 --model resnet
    ``` 
    ```python
    import PinkBlack.io
    # Set default argument
    args = PinkBlack.io.setup(default_args={'gpu':3,
                                            'batch_size':32,
                                            'model':"PinkNet"})
    # Autolog and pdb hooking will be activated.
    print(args.batch_size)
    ```
    ```bash
    64
    ```
4. 이미지 시각화 (Image Visualization)
5. Midi 객체 관련 함수 (Functions to Use Midi Instances)

> TODO :: Keras같은 Pytorch모델 훈련 인터페이스 