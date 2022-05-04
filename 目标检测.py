import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input,Lambda,Conv2D
from keras.models import load_model,Model
from yad2k.models.keras_yolo import yolo_head,yolo_boxes_to_corners,preprocess_true_boxes,yolo_loss,yolo_body
import yolo_utils

def yolo_filter_boxes(box_pc,boxes,box_class,threshold=0.0):
    box_score=box_pc*box_class
    box_class=K.argmax(box_score,axis=-1)
    box_class_score=K.max(box_score,axis=-1)

    filter_mask=(box_class_score>=threshold)
    score=tf.boolean_mask(box_class_score,filter_mask)
    boxes=tf.boolean_mask(boxes,filter_mask)
    classes=tf.boolean_mask(box_class,filter_mask)

    return score,boxes,classes

def iou(box1,box2):
    xi1=np.maximum(box1[0],box2[0])
    yi1=np.maximum(box1[1],box2[1])
    xi2=np.minimum(box1[3],box2[3])
    yi2=np.minimum(box1[3],box2[3])

    area=(xi1-xi2)*(yi1-yi2)
    box1_area=(box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area=(box2[2]-box2[0])*(box1[3]-box1[1])

    union_area=box1_area+box2_area-area
    iou=area/union_area
    return iou

def yolo_non_max_suppression(score,boxes,classes,max_boxes=10,iou_threshold=0.5):
    max=K.variable(max_boxes,dtype='int32')
    K.get_session().run(tf.variables_initializer([max]))
    nms_index=tf.image.non_max_suppression(boxes,score,max_boxes,iou_threshold)

    score=K.gather(score,nms_index)
    boxes=K.gather(boxes,nms_index)
    classes=K.gather(classes,nms_index)
    return  score,boxes,classes

def yolo_eval(yolo_outputs,image_shape=(720.,1280.),
              max_boxes=10,score_threshold=0.6,iou_threshold=0.5):
    box_pc,box_xy,box_wh,box_class=yolo_outputs
    boxes=yolo_boxes_to_corners(box_xy,box_wh)
    score,boxes,classes=yolo_filter_boxes(box_pc,boxes,box_class,score_threshold)
    boxes=yolo_utils.scale_boxes(boxes,image_shape)
    score,boxes,classes=yolo_non_max_suppression(score,boxes,classes,max_boxes,iou_threshold)

    return score,boxes,classes

def predict(sess,image_file,is_show=True,is_plot=True):
    image, image_data = yolo_utils.preprocess_image("images/" + image_file, model_image_size=(608, 608))
    out_score,out_boxes,out_classes=sess.run([score,boxes,classes],
                                             feed_dict={yolo_model.input:image_data,K.learning_phase():0})
    if is_show:
        print("在" + str(image_file) + "中找到了" + str(len(out_boxes)) + "个锚框。")

    color=yolo_utils.generate_colors(class_names)
    yolo_utils.draw_boxes(image,out_score,out_boxes,out_classes,class_names,color)
    image.save(os.path.join('out',image_file),quality=100)

    if is_plot:
        output_image = scipy.misc.imread(os.path.join("out", image_file))
        plt.imshow(output_image)

    return out_score,out_boxes,out_classes

sess=K.get_session()
class_names = yolo_utils.read_classes("model_data/coco_classes.txt")
anchors = yolo_utils.read_anchors("model_data/yolo_anchors.txt")
image_shape = (720.,1280.)
yolo_model=load_model('model_data/yolov2.h5')
yolo_outputs=yolo_head(yolo_model.output,anchors,len(class_names))
score,boxes,classes=yolo_eval(yolo_outputs,image_shape)

for i in range(1, 121):
    # 计算需要在前面填充几个0
    num_fill = int(len("0000") - len(str(1))) + 1
    # 对索引进行填充
    filename = str(i).zfill(num_fill) + ".jpg"
    print("当前文件：" + str(filename))

    # 开始绘制，不打印信息，不绘制图
    out_scores, out_boxes, out_classes = predict(sess, filename, is_show=True, is_plot=True)

print("绘制完成！")

out_scores, out_boxes, out_classes = predict(sess, "test.jpg")
plt.show()