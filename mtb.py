import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def read_files(dir_name = "./exposures"):
    images = []
    for filename in np.sort(os.listdir(dir_name)):
        if os.path.splitext(filename)[1] in ['.png', '.jpg']: # Only read png or jpg files
            img = cv2.imread(os.path.join(dir_name, filename))
            images.append(img)
    return images

def compute_bitmaps(img):
    height, width = img.shape

    t_bitmap = np.zeros((height,width), np.bool_)
    e_bitmap = np.zeros((height,width), np.bool_)
    median = np.median(img)
    
    for h in range(height):
        for w in range(width):
            if img[h,w] > median:
                t_bitmap[h,w] = True
            else:
                t_bitmap[h,w] = False
            if img[h,w] > median - 4 and img[h,w] < median + 4 :
                e_bitmap[h,w] = False
            else: 
                e_bitmap[h,w] = True    
           
    return t_bitmap, e_bitmap


def shift_bitmap(bitmap, shift_x, shift_y):
    # shift up/down
    bitmap = np.roll(bitmap, shift_y, axis=0)
    if shift_y > 0:
        bitmap[:shift_y] = np.zeros(bitmap[:shift_y].shape)
    elif shift_y < 0:
        bitmap[shift_y:] = np.zeros(bitmap[shift_y:].shape)
    # shift left/right
    bitmap = np.roll(bitmap, shift_x, axis=1)
    if shift_x > 0:
        bitmap[:, :shift_x] = np.zeros(bitmap[:,:shift_x].shape)
    elif shift_x < 0:
        bitmap[:, shift_x:] = np.zeros(bitmap[:,shift_x:].shape)

    return bitmap


# find shift_x and shift_y to move img2 so that it is aligned with img1
def find_shift(img1, img2, cur_shift_x, cur_shift_y):
    best_shift_x, best_shift_y = 0, 0
    t_bitmap1, e_bitmap1 = compute_bitmaps(img1) # t_bitmap: threshold bitmap
    t_bitmap2, e_bitmap2 = compute_bitmaps(img2) # e_bitmap: exclusion bitmap
    min_err = img1.shape[0] * img1.shape[1]

    for sx in [0,-1,1]:
        for sy in [0,-1,1]:
            shift_x = cur_shift_x + sx
            shift_y = cur_shift_y + sy

            t_bitmap2_shift = shift_bitmap(t_bitmap2, shift_x, shift_y)
            e_bitmap2_shift = shift_bitmap(e_bitmap2, shift_x, shift_y)

            diff_bitmap = np.logical_xor(t_bitmap1, t_bitmap2_shift)
            diff_bitmap = np.bitwise_and(diff_bitmap, e_bitmap1)
            diff_bitmap = np.bitwise_and(diff_bitmap, e_bitmap2_shift)

            err = np.sum(diff_bitmap)
            if err < min_err:
                min_err = err
                best_shift_x, best_shift_y = shift_x, shift_y
              
    return best_shift_x, best_shift_y


def get_exp_shift(img1, img2, max_shrink=3):
    height, width = img1.shape

    if max_shrink > 0:
        img1_shrink = cv2.resize(img1, (width//2, height//2))
        img2_shrink = cv2.resize(img2, (width//2, height//2))
        cur_shift_x, cur_shift_y = get_exp_shift(img1_shrink, img2_shrink, max_shrink-1)
        cur_shift_x *= 2
        cur_shift_y *= 2
    else:
        cur_shift_x, cur_shift_y = 0, 0
        
    shift_x, shift_y = find_shift(img1, img2, cur_shift_x, cur_shift_y)
    return shift_x, shift_y
    

def show_bitmap(bitmap):
    blank_image = np.zeros((bitmap.shape), np.uint8)
    for i in range(bitmap.shape[0]):
        for j in range(bitmap.shape[1]):
            if(bitmap[i,j] == False):
                blank_image[i,j] = 0
            else: 
                blank_image[i,j] = 255
    cv2.imshow("window",blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def translate(img, x, y):
    M = np.float32([[1,0,x],[0,1,y]])
    img_ = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_

def mtb(images):
    images_gray = []
    #convert into grayscale
    for img in images:
        img_gray = img[:,:,1] #BGR
        images_gray += [img_gray]

    #sample_i = len(images_gray)//2 # choose the middle image as sample image
    sample_i = 0
    
    for i in range(len(images_gray)):
        if i != sample_i :
            sx, sy = get_exp_shift(images_gray[sample_i], images_gray[i])
            print(i)
            print(sx, sy)
            print("--")
            images[i] = translate(images[i], sx, sy)

    return images

