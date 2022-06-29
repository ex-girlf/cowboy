from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
"""
python的用法 -》tensor数据类型
通过transforms.ToTensor去看两个问题
1.transform该如何使用
2.为什么需要tensor的数据类型

"""

#绝对路径 /Users/wudi/PycharmProjects/pythonProject/data/train/ants_image/5650366_e22b7e1065.jpg
#相对路径 data/train/ants_image/5650366_e22b7e1065.jpg

img_path = "data/train/ants_image/5650366_e22b7e1065.jpg"
img = Image.open(img_path)
tensor_pic = transforms.ToTensor()
tensor_img = tensor_pic(img)

write = SummaryWriter("log")
tran_tensor = transforms.Normalize([1, 2, 3], [0.5, 1, 3])
tran_norm = tran_tensor(tensor_img)
write.add_image("tran_tensor", tran_norm)
#write.add_image("Tensor_img", tensor_img)



write.close()



#print(tensor_img)
