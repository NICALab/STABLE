from torchvision import transforms
import torch
import random

grayscale = transforms.Grayscale(num_output_channels=1)
def overlay_images(pred, gt):
    while(len(pred.shape) < 4):
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    _, _, h, w = pred.shape
    overlay_img = torch.zeros((1, 3, h, w))

    overlay_img[0, 0, :, :] = pred
    overlay_img[0, 1, :, :] = gt

    return overlay_img

def random_rotate(img):
    rotation_angle = random.choice([90, 180, 270, 360])
    img = transforms.functional.rotate(img, rotation_angle)
    return img, rotation_angle

def random_flip(img):
    flip_key = (False, False)
    if random.random() < 0.5:
        img = transforms.functional.hflip(img)
        flip_key = (True, flip_key[1])
    if random.random() < 0.5:
        img = transforms.functional.vflip(img)
        flip_key = (flip_key[0], True)
    return img, flip_key