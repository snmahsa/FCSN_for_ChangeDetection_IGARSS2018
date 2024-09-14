import numpy as np
import os

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from model import FC_EF
form module import display_images
class OSCD(Dataset):
    def __init__(self, dir_nm):
        super(OSCD, self).__init__()
        self.dir_nm = dir_nm
        self.file_ls = os.listdir(dir_nm)
        self.file_size = len(self.file_ls)

    def __getitem__(self, idx):
        mat = np.load(self.dir_nm + self.file_ls[idx]).astype(np.float)
        x1 = mat[:3,:,:]/255
        x2 = mat[3:6,:,:]/255
        lbl = mat[6,:,:]/255
        return x1, x2, lbl

    def __len__(self):
        return self.file_size

def main():
    train_dir = './data/train/'
    test_dir = './data/test/'
    lr = 0.001
    test_all_preds = []
    test_all_labels = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = OSCD(train_dir)
    train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_data = OSCD(test_dir)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    model = FC_EF().to(device, dtype=torch.float)

    # optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    weights = torch.tensor([1.0, 20.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight =weights)

    for epoch in range(70):
        print(f"-----------------------epoch {epoch} :-----------------------")
        loss_v = []
        model.train()
        for i, data in enumerate(train_dataloader):
            x1, x2, lbl = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            lbl = lbl.to(device, dtype=torch.long)
            y = model(x1,x2)
            predicted_labels = torch.argmax(y, dim=1)
            if i < 2:
              print("Dimensions of x1:", x1.shape)
              print("Dimensions of x2:", x2.shape)
              print("Dimensions of lbl:", lbl.shape)
              print("Dimensions of y:", y.shape)
              print(f"sample {i} :")
              display_images(x1[0], x2[0], lbl[0], predicted_labels[0])
            optimizer.zero_grad()
            loss = criterion(y, lbl)
            loss.backward()
            optimizer.step()
            loss_v.append(loss.item())
            if(i%20==0 and i>0):
                print(f"---loss in {i/20} :",np.mean(loss_v))
                loss_v = []

    loss_v = []
    model.eval()
    test_all_preds.clear()
    test_all_labels.clear()

    with torch.no_grad(): 
        for i, data in enumerate(test_dataloader):
            x1, x2, lbl = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            lbl = lbl.to(device, dtype=torch.long)


            y = model(x1, x2)

            predicted_labels_test = torch.argmax(y, dim=1)
            predicted_labels_np = predicted_labels_test.cpu().numpy().flatten()
            lbl_np = lbl.cpu().numpy().flatten()

             
            test_all_labels.extend(lbl_np)
            test_all_preds.extend(predicted_labels_np)

            if i < 2:
                print(f"----test {i} :----")
                display_images(x1[0], x2[0], lbl[0], predicted_labels_test[0])

        precision = precision_score(test_all_labels, test_all_preds, average='macro')
        recall = recall_score(test_all_labels, test_all_preds, average='macro')
        f1 = f1_score(test_all_labels, test_all_preds, average='macro')

        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        print("-------------------------------")
    torch.save(model.state_dict(), f'model_weights_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()