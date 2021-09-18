from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

train = ImageFolder('/content/drive/MyDrive/projects/Dataset/training',transform=trans)
test = ImageFolder('/content/drive/MyDrive/projects/Dataset/validation',transform=trans)

train_dl = DataLoader(train,batch_size=32, shuffle=True)
test_dl = DataLoader(test,batch_size=32,shuffle=True)