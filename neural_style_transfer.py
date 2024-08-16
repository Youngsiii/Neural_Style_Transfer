import os
import time
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = ['0', '5', '10', '19', '28']  # take out feature maps in VGG19
        self.model = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features[
                     :29]  # features[0,1,2,...,28]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.features:
                features.append(x)
        return features


im_height = 400
im_width = 640
transform = transforms.Compose([
    transforms.Resize((im_height, im_width)),
    transforms.ToTensor(),  # image -> tensor
    # transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),

])

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(image_name):
    image = Image.open(image_name)  # PIL image
    image = transform(image).unsqueeze(0)  # PIL image -> tensor and (C, H, W) -> (1, C, H, W) add BS dim for VGG input
    return image.to(device)


content_img = load_image(r"/kaggle/input/nst-img/zurich.jpeg")  # content image   (1, 3, H, W)
style_img = load_image(r"/kaggle/input/nst-img/starry_night.jpg")  # style image     (1, 3, H, W)
# generated = torch.randn(content_img.shape, device=device, requires_grad=True)      # generated image (1, 3, H, W)

# 将generated初始化为content_img会产生更好的效果
generated = content_img.clone().requires_grad_(True)
# 使用.clone()创建content_img的副本，它们具有相同的数据，但是具有独立的计算历史，并且新创建的张量会被用于新的计算图
# .requires_grad_(True)  会使用该张量一起构建计算图，反向传播时会计算这个张量的梯度并更新这个张量的值


# hyperparameters
total_steps = 6001  # number of steps for updating the generated image
learning_rate = 0.001
alpha = 1  # weight for content loss
beta = 0.02  # weight for style loss
save_image_path = "/kaggle/working/"  # the file path for saving generated image

model = VGG().to(device).eval()  # freeze VGG and just update generated image
optimizer = torch.optim.Adam([generated], lr=learning_rate)

start = time.time()
for step in range(total_steps):

    # initialize loss for each step
    content_loss = 0
    style_loss = 0

    generated_features = model(generated)  # [gen_f1, gen_f2, gen_f3, gen_f4, gen_f5]
    content_features = model(content_img)  # [con_f1, con_f2, con_f3, con_f4, con_f5]
    style_features = model(style_img)  # [sty_f1, sty_f2, sty_f3, sty_f4, sty_f5]

    # compute loss for each feature
    # gen_feature/cont_feature/style_feature: (batch_size, channel, height, width)
    for gen_feature, cont_feature, style_feature in zip(generated_features, content_features, style_features):
        content_loss += torch.mean((gen_feature - cont_feature) ** 2)  # content loss

        # compute Gram matrix for gen_feature
        # Gram矩阵的作用是通过计算每一对特征图通道之间的内积来捕捉这些通道之间的相关性，因此Gram矩阵反映了图像的风格信息，忽略了空间结构
        batch_size, channel, height, width = gen_feature.shape  # batch_size = 1
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t())  # Gram matrix for gen_feature
        S = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t())  # Gram matrix for style_feature

        style_loss += torch.mean((G - S) ** 2)  # style loss

    total_loss = alpha * content_loss + beta * style_loss

    optimizer.zero_grad()  # zero the gradient
    total_loss.backward()  # compute gradient for updating generated image
    optimizer.step()  # update the generated image

    if step % 600 == 0:
        print(total_loss.item())
        save_image(generated, os.path.join(save_image_path, f"generated_{step}_img.jpg"))

end = time.time()
print(f"the time used is {end - start}")
















