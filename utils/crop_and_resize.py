def main(i):

    # from PIL import Image
    import cv2
    import os

    image_root_path = '../data/train/images_sub/'
    label_root_path = '../data/train/labels_sub/'

    image_list = os.listdir(image_root_path)
    label_list = os.listdir(label_root_path)

    image = cv2.imread(os.path.join(image_root_path, image_list[i]))
    label = cv2.imread(os.path.join(label_root_path, label_list[i]))

    w, h, c = image.shape

    image = cv2.resize(image, (w//2, h//2))
    label = cv2.resize(label, (w//2, h//2))

    # image = image.resize((w//2, h//2))
    # label = label.resize((w//2, h//2))

    save_root_1 = '../data/train/images_sub_256/'
    save_root_2 = '../data/train/labels_sub_256/'
    if not os.path.exists(save_root_1):
        os.makedirs(save_root_1)
    if not os.path.exists(save_root_2):
        os.makedirs(save_root_2)
    image_save_name = os.path.join(save_root_1, image_list[i])
    label_save_name = os.path.join(save_root_2, label_list[i])
    cv2.imwrite(image_save_name, image)
    cv2.imwrite(label_save_name, label)
    # image.save(image_save_name)
    # label.save(label_save_name)


if __name__ == '__main__':

    import multiprocessing
    import os

    image_root_path = '../data/train/images_sub/'
    label_root_path = '../data/train/labels_sub/'

    image_list = os.listdir(image_root_path)
    label_list = os.listdir(label_root_path)

    pool = multiprocessing.Pool(processes=6)
    pool.map(main, range(len(image_list)))
    pool.close()
    pool.join()




