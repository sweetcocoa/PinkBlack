# PinkBlack #
----
> 개발(주로 딥러닝)을 위한 여러 가지 잡 모듈

1. 자동 로깅 (Autolog)
    - stdout, stderr redirection
2. 에러 발생할 때 pdb hooking (Auto-raise pdb)
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
    ``
4. 이미지 시각화 (Image Visualization)

5. 훈련을 위한 Trainer 객체
    ```python
    import PinkBlack.trainer
    trainer = PinkBlack.trainer.Trainer(network, 
                                        train_dataloader=tdl, 
                                        val_dataloader=vdl,
                                        lr_scheduler=steplr,
                                        logdir="tensorboard_dir",
                                        ckpt="ckpt/trained.pth")
    trainer.load("ckpt/trained.pth")
    trainer.train(epoch=30)
    trainer.save("ckpt/trained_last.pth")
    ``` 
    - 1 epoch 단위로 validation
    - train loss, train metric, validation loss, validation metric을 tensorboard에 기록
    - max validation metric, current epoch 등을 ckpt+".config" 에 저장, 로드

6. face detection, facial landmark detection module

```python
import PinkBlack.face
sfd = PinkBlack.face.S3FD("s3fd.pth")
fan = FAN3D("3dfan.pth")

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