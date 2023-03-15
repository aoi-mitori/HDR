import cv2
import numpy as np
import os

# exclusion bitmap
def compute_bitmap(img):
    height, width = img.shape

    bitmap = np.zeros((height,width), np.bool_)
    median = np.median(img)
    
    for h in range(height):
        for w in range(width):
            if img[h,w] > median+4 :
                bitmap[h,w] = True

    return bitmap

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
        bitmap[:,:shift_x] = np.zeros(bitmap[:,:shift_x].shape)
    elif shift_x < 0:
        bitmap[:,shift_x:] = np.zeros(bitmap[:,shift_x:].shape)

    return bitmap

# find shift_x and shift_y to move img2 so that it is aligned with img1
# def find_shift(bitmap1, bitmap2):
#     for shift_x in [-1,0,1]:
#         for shift_y in [-1,0,1]:


    
def get_exp_shift(img1, img2, max_shrink=3):
    height, width = img1.shape
    # print(max_shrink)
    # print(img1.shape)
    # print("==")
    
    if max_shrink > 0:
        img1_shrink = cv2.resize(img1, (width//2, height//2))
        img2_shrink = cv2.resize(img2, (width//2, height//2))
        get_exp_shift(img1_shrink, img2_shrink, max_shrink-1)
    
    print(max_shrink)
    print(img1.shape)
    print("--")

    bitmap1 = compute_bitmap(img1)
    bitmap2 = compute_bitmap(img2)

    

    

    




dir_name = "./Memorial_SourceImages"
images = []
images_gray = []

for filename in np.sort(os.listdir(dir_name)):
    if filename.split(".")[1] in ['png']:
        img = cv2.imread(os.path.join(dir_name, filename))
        images += [img]

#convert into grayscale
for img in images:
    img_gray = img[:,:,1] #BGR
    images_gray += [img_gray]

get_exp_shift(images_gray[0],images_gray[1])



# blank_image = np.zeros((bitmap.shape), np.uint8)
# for i in range(bitmap.shape[0]):
#     for j in range(bitmap.shape[1]):
#         if(bitmap[i,j] == False):
#             blank_image[i,j] = 0
#         else: 
#             blank_image[i,j] = 255

# cv2.imshow("1",blank_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("1",images[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

