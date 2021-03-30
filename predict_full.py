import numpy as np
import torch
from PIL import Image
from unet import UNet
from torchvision import transforms
import os


def predict_img(net, full_img, device):
    net.eval()

    norm_img = transforms.Compose([
        transforms.Resize(256),  # DEBUG: predict 256 images
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    img = norm_img(full_img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        # print(torch.max(output), torch.min(output))
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
    return full_mask


def get_iou(source, target, thresh=0.5):
    source = np.array(source)
    target = np.array(target)

    source = source / 255
    target = target / 255

    source = (source > thresh) * 1
    target = (target > 0.5) * 1

    inter = np.logical_and(source, target)
    union = np.logical_or(source, target)

    iou = np.sum(inter) / np.sum(union)

    return iou, source*255


def main(fn_image, fn_label):
    resize_h = 3072
    resize_w = 4608
    sub_size = 512
    stride = 256
    overlap = sub_size - stride

    n_row = 11
    n_col = 17

    image = Image.open(fn_image)
    label = Image.open(fn_label)

    full_w, full_h = image.size

    image_resize = image.resize((resize_w, resize_h))
    image_array = np.array(image_resize)

    prediction = np.zeros((2, resize_h, resize_w))

    for c in range(n_col):
        for r in range(n_row):
            sub_image = image_array[r*stride:r*stride+sub_size, c*stride:c*stride+sub_size]
            sub_image = Image.fromarray(sub_image.astype(np.uint8))

            mask_soft = predict_img(net=net,
                                    full_img=sub_image,
                                    device=device)

            mask_soft = np.array(mask_soft * 255, dtype=int)

            if c%2 and r%2:
                prediction[1, r*stride:r*stride+sub_size, c*stride:c*stride+sub_size] = mask_soft
            else:
                prediction[0, r * stride:r * stride + sub_size, c * stride:c * stride + sub_size] = mask_soft

    prediction = np.mean(prediction, axis=0)
    prediction = Image.fromarray(prediction.astype(np.uint8))
    prediction = prediction.resize((full_w, full_h))

    iou, prediction_binary = get_iou(prediction, label)
    prediction_binary = Image.fromarray(prediction_binary.astype(np.uint8))

    fn_save = fn_label.replace('.png', '_pred.png')
    prediction_binary.save(fn_save)

    print('Result has been saved to %s' % fn_save)
    print('IoU=%.4f' % iou)
    print()

    return iou


if __name__ == "__main__":

    net = UNet(n_channels=3, n_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load('./checkpoints/CP_full.pth', map_location=device))

    root_dir_image = './_raw_data/val/images/'
    root_dir_label = './_raw_data/val/labels_binary/'

    # ### single image
    #
    # id = 162
    #
    # fn_image = os.path.join(root_dir_image, '%s.png' % id)
    # fn_label = os.path.join(root_dir_label, '%s.png' % id)
    #
    # main(fn_image, fn_label)


    ### multiple images

    ids = range(161, 166)
    iou_list = []

    for id in ids:

        fn_image = os.path.join(root_dir_image, '%s.png' % id)
        fn_label = os.path.join(root_dir_label, '%s.png' % id)

        iou = main(fn_image, fn_label)

        iou_list.append(iou)

    miou = np.mean(iou_list)
    print('mIoU=%4f' % miou)
