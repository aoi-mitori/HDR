import numpy as np

# bitmap = np.zeros((3,3), np.bool_)
# bitmap[2,2] = True
# bitmap[0,2] = True

# print(bitmap)
# print("----")

# # bitmap_0 = np.roll(bitmap,1,axis=0) # y axis
# # print(bitmap_0)
# # print("---")
# # print(bitmap_0[:2])
# # print(bitmap_0[:2].shape)
# # print("+++++")
# # bitmap_0[:1] = np.zeros((bitmap_0[:1].shape))

# # print(bitmap_0)
# # print("-----")

# bitmap_1 = np.roll(bitmap,1,axis=1) # x axis
# print(bitmap_1)
# print("---")
# print(bitmap_1[1:])

import numpy as np
a = np.array([[ 1, 2, 3],
              [ 4, 5, 6],
              [ 7, 8, 9],
              [10, 11, 12]])  
print(a)  


def shift_array(array, place):
    new_arr = np.roll(array, place, axis=0)
    if(place > 0):
        new_arr[:place] = np.zeros(new_arr[:place].shape)
    elif(place < 0):
        new_arr[place:] = np.zeros((new_arr[place:].shape))
    
    return new_arr
print(a[:,0])


# print(np.array(a[:,:1]))
# print(np.array(a[:,:1]).shape)
# # print(a[:,0])

# print(np.array(a[:,-1:]))


# a[:,[1]] = np.zeros(a[:,[1]].shape)
# print(a)


def shift_array(array, place):
    new_arr = np.roll(array, place, axis=1)
    if(place > 0):
        new_arr[:,:place] = np.zeros((new_arr[:,:place].shape))
    elif(place < 0):
        new_arr[:,place:] = np.zeros((new_arr[:,place:].shape))

    return new_arr


print(shift_array(a,2))
print(shift_array(a,-2))
