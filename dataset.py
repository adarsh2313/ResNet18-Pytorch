import cv2
import torch
import glob

# For X_train and Y_train : 

boots = glob.glob('/content/drive/MyDrive/projects/Dataset/training/boots/*.png')                     # 0 to 248
flipflops = glob.glob('/content/drive/MyDrive/projects/Dataset/training/flip_flops/*.png')            # 249 to 497
loafers = glob.glob('/content/drive/MyDrive/projects/Dataset/training/loafers/*.png')                 # 498 to 746
sandals = glob.glob('/content/drive/MyDrive/projects/Dataset/training/sandals/*.png')                 # 747 to 995
sneakers = glob.glob('/content/drive/MyDrive/projects/Dataset/training/sneakers/*.png')               # 996 to 1244

train_paths = boots+flipflops+loafers+sandals+sneakers
length = len(train_paths)

X_train = torch.zeros((length,3,224,224))

for i in range(length):
    img = cv2.imread(train_paths[i])
    img = cv2.resize(img,(224,224))
    img = torch.Tensor(img)/255
    img = img.reshape(3,224,224)
    X_train[i] = img

Y_train = torch.zeros((length,1,5))

for i in range(length):
    if i<=248:
        Y_train[i,0,0] = 1
    elif i>=249 and i<=497:
        Y_train[i,0,1] = 1
    elif i>=498 and i<=746:
        Y_train[i,0,2] = 1
    elif i>=747 and i<=995:
        Y_train[i,0,3] = 1
    else:
        Y_train[i,0,4] = 1

# For X_test and Y_test : 

t_boots = glob.glob('/content/drive/MyDrive/projects/Dataset/validation/boots/*.png')                     # 0 to 49
t_flipflops = glob.glob('/content/drive/MyDrive/projects/Dataset/validation/flip_flops/*.png')            # 50 to 99
t_loafers = glob.glob('/content/drive/MyDrive/projects/Dataset/validation/loafers/*.png')                 # 100 to 149
t_sandals = glob.glob('/content/drive/MyDrive/projects/Dataset/validation/sandals/*.png')                 # 150 to 199
t_sneakers = glob.glob('/content/drive/MyDrive/projects/Dataset/validation/sneakers/*.png')               # 200 to 249

test_paths = t_boots+t_flipflops+t_loafers+t_sandals+t_sneakers
t_length = len(test_paths)

X_test = torch.zeros((t_length,3,224,224))

for i in range(t_length):
    img = cv2.imread(test_paths[i])
    img = cv2.resize(img,(224,224))
    img = torch.Tensor(img)/255
    img = img.reshape(3,224,224)
    X_test[i] = img

Y_test = torch.zeros((t_length,1,5))
for i in range(t_length):
    if i<=49:
        Y_test[i,0,0] = 1
    elif i>=50 and i<=99:
        Y_test[i,0,1] = 1
    elif i>=100 and i<=149:
        Y_test[i,0,2] = 1
    elif i>=150 and i<=199:
        Y_test[i,0,3] = 1
    else:
        Y_test[i,0,4] = 1

