
import os
import cv2
cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
import torch.nn.functional as F
from torch.serialization import SourceChangeWarning
import warnings



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flip_tensor_lr(batch):
    columns = batch.data.size()[-1]
    index = torch.autograd.Variable(torch.LongTensor(list(reversed(range(columns)))).to(device))
    return batch.index_select(3, index)


def flip_tensor_ud(batch):
    rows = batch.data.size()[-2]
    index = torch.autograd.Variable(torch.LongTensor(list(reversed(range(rows)))).to(device))
    return batch.index_select(2, index)


def to_numpy(batch):
    return np.moveaxis(batch.data.cpu().numpy(), 1, -1)


def _8bitGray2Input(img_arr):
    return np.repeat(np.expand_dims(img_arr, axis=2), 3, axis=2).astype(np.uint8)


def _16bitGray2Input(img_arr):
    return np.repeat(np.expand_dims(img_arr, axis=2) / 256, 3, axis=2).astype(np.uint8)


def _8bitRGB2Input(img_arr):
    return img_arr

def _12bitRGB2Input(img_arr):
    return (img_arr / 16).astype(np.uint8)


def img_to_tensor(im):
    # Check if batch (rank 4) or single image (rank 3)
    if len(im.shape) == 4:
        # Batch: (N, H, W, C) -> (N, C, H, W)
        return torch.from_numpy(np.moveaxis(im / (255. if im.dtype == np.uint8 else 1), -1, 1).astype(np.float32))
    else:
        # Single: (H, W, C) -> (1, C, H, W)
        return torch.from_numpy(np.expand_dims(np.moveaxis(im / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32), axis=0))


# Load all four models, return a list of these models.
def read_model(model_paths):
    # torch.cuda.clear_memory_allocated
    with warnings.catch_warnings():
        models = []
        for model_path in model_paths:
            print('start')
            warnings.simplefilter('ignore', SourceChangeWarning)
            if torch.cuda.is_available():
                model = torch.load(model_path, weights_only=False)
            else:
                model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            print(model_path)
#            for i, (name, module) in enumerate(model._modules.items()):
#                module = recursion_change_bn(model)
            model.eval()
            models.append(model)
        assert len(models) == 4
        return models

# Do prediction, take a (16-bit) 512*512 image numpy array, return a 512*512 8-bit image array with dtype=uint8.
# 16_bit_gray, 8_bit_gray, 8_bit_RGB
def predict(models, img_arr, image_type='12_bit_RGB'):
    # pdb.set_trace()
    conversion = {'16_bit_gray':_16bitGray2Input, '8_bit_gray':_8bitGray2Input, '8_bit_RGB':_8bitRGB2Input, '12_bit_RGB':_12bitRGB2Input}
    assert(image_type in conversion.keys())
    
    # Handle batch processing
    is_batch = len(img_arr.shape) == 4
    if is_batch:
        # Assume all images in batch are same type/shape
        # _8bitRGB2Input currently returns identity, so mapping directly works
        # If conversion functions change dimension, we'd need loop or vectorize them
        # For now, _8bitRGB2Input is just identity, so:
        rgb_img_arr = img_arr 
    else:
        rgb_img_arr = conversion[image_type](img_arr)
        
    with torch.no_grad():
        batch = torch.autograd.Variable(img_to_tensor(rgb_img_arr)).to(device)
        # print(batch.shape)
        ret_arr = []
        for model in models:
            # model.cuda()  ## added by pratik - moving both tensors in same gpu device
            if torch.cuda.is_available():
                 model.cuda()
            
            # Remove the first F.sigmoid because it will be applied later
            pred1 = model(batch)
            pred2 = flip_tensor_lr(model(flip_tensor_lr(batch)))
            pred3 = flip_tensor_ud(model(flip_tensor_ud(batch)))
            pred4 = flip_tensor_ud(flip_tensor_lr(model(flip_tensor_ud(flip_tensor_lr(batch)))))

            masks = [pred1, pred2, pred3, pred4]
            masks = list(map(F.sigmoid, masks))
            new_mask = torch.mean(torch.stack(masks, 0), 0) # Shape: (N, 1, H, W)
            ret_arr.append(to_numpy(new_mask)) # to_numpy moves axis 1 to -1 -> (N, H, W, 1) or (1, H, W, 1)
            
            del pred1
        del pred2
        del pred3
        del pred4
        del masks
        del new_mask
        # del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    merged = np.mean(ret_arr, axis=0) # Shape: (N, H, W, 1)
    merged = np.squeeze(merged) # Shape: (N, H, W) or (H, W) if N=1 and squeeze handles it... 
                                # Wait, np.squeeze might remove batch dim if N=1. 
                                # Original code: np.squeeze((1, H, W, 1)) -> (H, W). Correct.
                                # Batch: np.squeeze((N, H, W, 1)) -> (N, H, W). Correct.
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (merged * 255).astype(np.uint8)

















# numpy array?
# def img2RGB(img):
#     pixels = img.load()
#     for i in range(img.size[0]):
#         for j in range(img.size[1]):
#             pixels[i, j] = pixels[i, j] // 256
#     rgbimg = Image.new('RGBA', img.size)
#     rgbimg.paste(img)
#     return np.asarray(rgbimg)[..., :-1]

# demo for test
# if __name__ == '__main__':
    # model_paths = ['fold{}_best.pth'.format(i) for i in range(4)]
    # img_path = 'Sec131X2574Y2906_row.tif'
    # output_path = 'output.tif'
    # print(model_paths)
    # img = Image.open(img_path)
    # img = img_to_tensor(img2RGB(img))
    # # img = img_to_tensor(imread(img_path, mode='RGB'))
    #
    # # print(img.cpu().numpy())
    # models = read_model(model_paths)
    # ret = predict(models, img)
    # # print(ret)
    # ret = np.squeeze(ret)
    # final_ret = (ret * 255).astype(np.uint8)
    # cv2.imwrite(output_path, final_ret)
