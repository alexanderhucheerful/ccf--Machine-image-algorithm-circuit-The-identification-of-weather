class DefaultConfigs(object):
    #1.string parameters
    train_data = "D:/metedata/Train_label.csv"
    root1 = "D:/metedata/Train/"
    test_data = "D:/metedata/Train/"
    val_data = "/home/user/zcj/tutorials/data/val/"
    model_name = "resnet50"
    weights = "C:/Users/alexanderhu/Desktop/pytorch-image-classification-master/checkpoints/"
    best_models = weights + "best_model/"
    submit = "C:/Users/alexanderhu/Desktop/pytorch-image-classification-master/submit/"
    logs = "C:/Users/alexanderhu/Desktop/pytorch-image-classification-master/logs/"
    gpus = "1"

    #2.numeric parameters
    epochs = 10
    batch_size = 16
    img_height = 300
    img_weight = 300
    num_classes = 10
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()
