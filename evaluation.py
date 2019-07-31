import os
import glob
import argparse

import keras
import tensorflow as tf
from keras.models import load_model

from keras_contrib.layers import InstanceNormalization

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def get_config(mode):
    config = {
        "1": { # 1st cascade
            'checkpoint': './checkpoint/model0.h5'
            }, 
        "2_1": {
            'checkpoint': './checkpoint/model1.h5'
            },
        "2_2": {
            'checkpoint': './checkpoint/model2.h5',
            'lossfn': 'dice',
            'depth': 4,
            'standard': 'normal',
            'task': 'tumor',
            'wlevel': 100,
            'wwidth': 400
            },
        "2_3": {
            'checkpoint': './checkpoint/model3.h5',
            'lossfn': 'dice',
            'depth': 3,
            'standard': 'minmax',
            'task': 'tumor1',
            'wlevel': 100,
            'wwidth': 400
            },
        "2_4": {
            'checkpoint': './checkpoint/model4.h5',
            'lossfn': 'focaldice',
            'depth': 3,
            'standard': 'minmax',
            'task': 'tumor1',
            'wlevel': 100,
            'wwidth': 400
            },
        "2_5": {
            'checkpoint': './checkpoint/model5.h5',
            'lossfn': 'dice',
            'depth': 3,
            'standard': 'normal',
            'task': 'tumor1',
            'wlevel': 100,
            'wwidth': 400
            }}

    return config[mode]

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None, metavar="1 / 2_1 / 2_2 / 2_3 / 2_4 / 2_5")
    parser.add_argument("--testset", type=str, default=None, metavar="/path/testset")

    return parser.parse_args()

def main():
    args = get_arguments()
    assert args.mode
    assert args.testset

    keras.backend.tensorflow_backend.set_session(get_session())

    if not os.path.isdir('./result'):
        os.mkdir('./result')
    if not os.path.isdir(os.path.join('./result', args.mode)):
        os.mkdir(os.path.join('./result', args.mode))

    testlist = sorted([os.path.join(args.testset, d) for d in os.listdir(args.testset) if 'case' in d])    
    config = get_config(args.mode)

    if args.mode == '1':
        from cascade_1st.Model_ACE_CNet import load_model
        model = load_model(
            input_shape=(200, 200, 200, 1), 
            num_labels=1, 
            base_filter=32,
            depth_size=3, 
            se_res_block=True, 
            se_ratio=16, 
            last_relu=True
            )

        model.load_weights(config['checkpoint'])
        
    else:
        if args.mode == '2_1':
            from cascade_2nd.model_1.Model_ACE_CNet_2ndstg import load_model
            model = load_model(
                input_shape=(200, 200, 200, 1), 
                num_labels=3, 
                base_filter=32,
                depth_size=3, 
                se_res_block=True, 
                se_ratio=16, 
                last_relu=False
                )

            model.load_weights(config['checkpoint'])
            
        else:
            from cascade_2nd.model_2_5.model import MyModel
            from cascade_2nd.model_2_5.load_data import Preprocessing

            model = MyModel(
                model=args.mode,
                input_shape=(None, None, None, 1),
                lossfn=config['lossfn'],
                classes=3,
                depth=config['depth']
                )

            model.mymodel.load_weights(config['checkpoint'])

            prep = Preprocessing(
                task=config['task'],
                standard=config['standard'],
                wlevel=config['wlevel'],
                wwidth=config['wwidth'],
                rotation_range=[0., 0., 0.]
                )
    

    if args.mode in ['1', '2_1']:

    else:
        loop = 2 if TASK == 'tumor' else 1
        for i, data in enumerate(testlist):
            print(data)
            img_orig = sitk.ReadImage(os.path.join(data, 'imaging.nii'))
            mask_orig = sitk.ReadImage(os.path.join('./result/1', data+'.nii'))

            result_save = np.zeros_like(sitk.GetArrayFromImage(mask_orig))
            for idx in range(loop):
                img, mask, spacing = prep._array2img([img_orig, mask_orig], True)
                img, mask, flag, bbox, diff, diff1 = prep._getvoi([img, mask, idx, True)
                if flag:
                    if idx == 1 and TASK == 'tumor':
                        img, mask = prep._horizontal_flip([img, mask])
                        
                    img = prep._windowing(img)
                    img = prep._standard(img)
                    mask = prep._onehot(mask)
                    img, mask = prep._expand([img, mask])
            
                    result = model.mymodel.predict_on_batch(img)
                    result = np.argmax(np.squeeze(result), axis=-1)
                    label = np.argmax(np.squeeze(mask), axis=-1)
                    
                    if TASK == 'tumor':
                        if idx == 1:
                            img, result = prep._horizontal_flip([img, result])
                        result_save[np.maximum(0, bbox[0]-margin):np.minimum(result_save.shape[0]-1, bbox[1]+1+margin),
                                    np.maximum(0, bbox[2]-margin):np.minimum(result_save.shape[1]-1, bbox[3]+1+margin),
                                    np.maximum(0, bbox[4]-margin):np.minimum(result_save.shape[2]-1, bbox[5]+1+margin)] = result
                    
                    elif TASK == 'tumor1':
                        threshold = [380, 230, 72]
                        mask_orig = sitk.GetArrayFromImage(mask_orig)
                        result_save[np.maximum(0,bbox[0]):np.minimum(result_save.shape[0],bbox[1]),
                                    np.maximum(0,bbox[2]):np.minimum(result_save.shape[1],bbox[3]),
                                    np.maximum(0,bbox[4]):np.minimum(result_save.shape[2],bbox[5])] = result[diff[0]//2:-diff[0]//2-diff1[0] if -diff[0]//2-diff1[0] != 0 else result.shape[0],
                                                                                                            diff[1]//2:-diff[1]//2-diff1[1] if -diff[1]//2-diff1[1] != 0 else result.shape[1],
                                                                                                            diff[2]//2:-diff[2]//2-diff1[2] if -diff[2]//2-diff1[2] != 0 else result.shape[2]]
                    
            temp2 = np.swapaxes(result_save, 1, 2)
            temp2 = np.swapaxes(temp2, 0, 1)
            temp2 = np.swapaxes(temp2, 1, 2)

            img_pair = nib.Nifti1Pair(temp2, np.diag([-spacing[0], spacing[1], spacing[2], 1]))
            nib.save(img_pair, os.path.join('./result', args.mode, data+'.nii'))

if __name__ == "__main__":
    main()