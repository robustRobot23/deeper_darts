from ultralytics import YOLO

import os
import os.path as osp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from yacs.config import CfgNode as CN
from dataloader import load_tfds
import numpy as np
import argparse
from utils import detect_hardware
import pickle

import random

# # Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
# results = model.train(data='coco128.yaml', epochs=100, imgsz=800)

def train(cfg, strategy):
    img_path = osp.join(cfg.data.path, 'cropped_images', str(cfg.model.input_size))
    assert osp.exists(img_path), 'Could not find cropped images at {}'.format(img_path)

    tf.random.set_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)

    with strategy.scope():
        yolo = build_model(cfg)

    if cfg.model.weights_path:
        if cfg.model.weights_path.endswith('.h5'):
            yolo.model.load_weights(cfg.model.weights_path, by_name=True, skip_mismatch=True)
        else:
            if 'weights_layers' in cfg.model:
                pretrained_model = build_model(cfg).model
                pretrained_model.load_weights(cfg.model.weights_path)
                for module, pretrained_module in zip(yolo.model.layers, pretrained_model.layers):
                    for layer, pretrained_layer in zip(module.layers, pretrained_module.layers):
                        if layer.name in cfg.model.weights_layers:
                            layer.set_weights(pretrained_layer.get_weights())
                            print('Transferred pretrained weights to', layer.name)
                del pretrained_model
            else:
                yolo.load_weights(
                    weights_path=cfg.model.weights_path,
                    weights_type=cfg.model.weights_type)

    yolo_dataset_object = yolo.load_dataset('dummy_dataset.txt', label_smoothing=0.)
    bbox_to_gt_func = yolo_dataset_object.bboxes_to_ground_truth

    train_ds = load_tfds(
        cfg,
        bbox_to_gt_func,
        split='train',
        batch_size=cfg.train.batch_size * strategy.num_replicas_in_sync)

    val_ds = load_tfds(
        cfg,
        bbox_to_gt_func,
        split='val',
        batch_size=cfg.train.batch_size * strategy.num_replicas_in_sync)

    n_train = train_ds.__len__()
    n_val = val_ds.__len__()
    print('Train samples:', n_train)
    print('Val samples:', n_val)

    spe = int(np.ceil(n_train / (cfg.train.batch_size * strategy.num_replicas_in_sync)))

    with strategy.scope():
        lr = tf.keras.experimental.CosineDecay(cfg.train.lr, cfg.train.epochs * spe)
        optimizer = tf.keras.optimizers.Adam(lr)
        loss = YOLOv4Loss(
            batch_size=yolo.batch_size,
            iou_type=cfg.train.loss_type,
            verbose=cfg.train.loss_verbose)
        yolo.model.compile(optimizer=optimizer, loss=loss)

    val_steps = {'d1': 20, 'd2': 8}

    hist = yolo.model.fit(
        train_ds,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        verbose=cfg.train.verbose,
        validation_data=None if not cfg.train.val else val_ds,
        validation_steps=val_steps[cfg.data.dataset] // strategy.num_replicas_in_sync,
        steps_per_epoch=spe)

    yolo.save_weights(
        weights_path='./models/{}/weights'.format(cfg.model.name),
        weights_type=cfg.train.save_weights_type)

    pickle.dump(hist.history, open('./models/{}/history.pkl'.format(cfg.model.name), 'wb'))
    return yolo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='default')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    tpu, strategy = detect_hardware(tpu_name=None)
    yolo = train(cfg, strategy)
    # predict(yolo, cfg, dataset=cfg.data.dataset, split='val')