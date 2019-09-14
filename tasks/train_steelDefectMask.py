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

import csv
def log_to_csv(data, csv_file):
    with open(csv_file, "a") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()


log_to_csv([0,0,0], "logs/dummy.csv")
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
batch_size = 4
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
        #--- Train
        acc_loss = 0
        running_loss = []
        for ith, (img, gt) in enumerate(train_dataloader):
            img = img.to(device)
            gt = gt.to(device)
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
                    .format(epoch+1, num_epochs, (ith+1)//acc_batch, acc_loss.data))
                running_loss.append(acc_loss)
                acc_loss=0
        log_to_csv(running_loss, "logs/train_batchloss.csv")

        #--- Validate
        val_loss = 0
        for jth, (val_img, val_gt) in enumerate(test_dataloader):
            val_img = val_img.to(device)
            val_gt = val_gt.to(device)
            with torch.no_grad():
                val_output = model(val_img)
                val_loss += criterion(val_output, val_gt)
        val_loss = val_loss / len(test_dataloader)

        print('epoch[{}/{}], [-----TEST------] loss:{:.4f}'
              .format(epoch+1, num_epochs, val_loss.data))
        log_to_csv(val_loss, "logs/test_loss.csv")

        if val_loss < best_loss:
            print("***saving best optimal state [Loss:{}] ***").format(val_loss.data)
            best_loss = val_loss
            torch.save(model.state_dict(), "weights/model.pth")
