# PinkBlack #
----
> 딥러닝을 위한 여러 가지 잡 모듈

1. 자동 로깅 (Autolog)
    - stdout, stderr redirection

2. yaml 설정파일 입력, 커맨드라인을 통한 overriding
    ```yaml
    # setting.yaml
    batch_size: 32
    gpu: 3
    model: "ResNet"
    ```

    ```python
    import PinkBlack.io
    # Set default argument
    args = PinkBlack.io.setup(default_args="setting.yaml") # 또는
    args = PinkBlack.io.setup(default_args=dict(gpu=3,
                                            batch_size=32,
                                            model="ResNet"))
    print(args.batch_size)
    ```
    ```bash
    $ python main.py --gpu 2 --batch_size 64 --model resnet
    >>> 64
    ```

3. 훈련을 위한 Trainer 객체
    ```python
    import PinkBlack.trainer
    trainer = PinkBlack.trainer.Trainer(network, 
                                        train_dataloader=tdl, 
                                        val_dataloader=vdl,
                                        lr_scheduler=steplr,
                                        tensorboard_dir="tensorboard_dir",
                                        ckpt="ckpt/trained.pth")
    trainer.load("ckpt/trained.pth")
    trainer.train(epoch=100)
    trainer.save("ckpt/trained_last.pth")
    ``` 
    - 1 epoch 단위로 validation
    - train loss, train metric, validation loss, validation metric을 tensorboard에 기록
    - max validation metric, current epoch 등을 ckpt+".config" 에 저장, 로드

5. face detection, facial landmark detection module
    ```python
    import PinkBlack.face
    sfd = PinkBlack.face.S3FD("s3fd.pth")
    fan = PinkBlack.face.FAN3D("3dfan.pth")
    
    # detect face
    # input : bgr cv2 image [height, width, 3]
    # return : [[x1, y1, x2, y2, conf], ... ]
    boxes = sfd.get_face_boxes(img_bgr)
    
    box = boxes[0]
    
    # detect facial landmark
    # return : [[68, 2]] np array
    ldmk = fan.get_landmarks(img_bgr, box)[0].tolist()
    
    # get translation, rotation (t, r)
    # return : t, r
    # t : [x, y, z]
    # r : [x, y, z]
    tr = fan.get_tr(fov=40, np.array(ldmk), img_bgr.shape)
    ```