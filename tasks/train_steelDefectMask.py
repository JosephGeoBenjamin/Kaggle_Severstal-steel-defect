''' 4 class Mask Layers

Output: 4 binary mask layers each corresponding to a class

'''

import torch
from torch.utils.data import DataLoader
from utilities.severstalData_utils import SeverstalSteelData
from networks.resnet_unet import ResNet18UNet
import utilities.lossMetrics_utils as LossMet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#----

DATASET_PATH='datasets/severstal/'

train_dataset = SeverstalSteelData(csv_file='train.csv',
                                    root_dir= DATASET_PATH,
                                    device = device)
train_dataloader = DataLoader(train_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

test_dataset = SeverstalSteelData(csv_file='validate.csv',
                                   root_dir=DATASET_PATH,
                                   device = device)
test_dataloader = DataLoader(test_dataset, batch_size=4,
                        shuffle=False, num_workers=4)
#----

num_epochs = 10000
batch_size = 2
acc_batch = 16 / batch_size
learning_rate = 1e-4

model = ResNet18UNet(4).to(device)
# model.load_state_dict(torch.load("/content/conv_autoencoder.pth"))
criterion = LossMet.DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

#----

if __name__ =="__main__":
    best_loss = 0
    for epoch in range(num_epochs):
        acc_loss = 0
        for ith, img, gt in enumerate(train_dataloader):
            #--- forward
            output = model(img)
            loss = criterion(output, gt) / acc_batch
            acc_loss += loss
            #--- backward
            loss.backward()
            if ( (ith+1) % acc_batch == 0):
                optimizer.step()
                optimizer.zero_grad()
                print('epoch[{}/{}], Mini Batch-{} loss:{:.4f}'
                    .format(epoch+1, ith+1/acc_batch, num_epochs, acc_loss.data))
                acc_loss=0

        for jth, val_img, val_gt in enumerate(test_dataloader):
            val_loss = 0
            with torch.no_grad():
                val_output = model(val_img)
                val_loss += criterion(val_output, val_gt)
            val_loss = val_loss / len(test_dataloader)
            print('epoch [{}/{}], [-----TEST------] loss:{:.4f}'
              .format(epoch+1, num_epochs, val_loss.data))

            if val_loss < best_loss:
                print("***saving best optimal state [Loss:{}] ***").format(val_loss.data)
                best_loss = val_loss
                torch.save(model.state_dict(), "weights/model.pth")

