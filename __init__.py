# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import data_manipulation as dm
import data_visualization as dv
from dataset_class import MriDataset
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from torchmetrics import accuracy
from example_models import UNET
import torch.nn as nn


# from torchvision.models import resnet50, ResNet50_Weights
# from example_models import model_0

from training_utils import print_train_time
from tqdm.auto import tqdm

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO : aggiungere metodo per eliminare le slices vuote


from torch.utils import data

if __name__ == '__main__':


    # TODO: aggiungere controllo sull'esistenza del file
    # try:
    files_path = '/media/neuropsycad/disk12t/EdoardoFilippiMasterThesis/implementations/unetsv3/scripts/img_and_annot.json'
    # basepath = '/media/neuropsycad/disk12t/EdoardoFilippiMasterThesis/implementations/unetsv3/dataset2'

    dataset_train = MriDataset(files_path, 0, [0, 0.9])
    dataset_test = MriDataset(files_path, 0, [0.9,1])
    # dataset_len = len(dataset)

    # example = dataset_train[torch.randint(0, len(dataset_train), (1,)).item()]
    # dv.see_data_sample(example)


    dataloader_train = DataLoader(dataset_train, batch_size=4,
                            shuffle=True, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=4,
                            shuffle=False, num_workers=0)

    print(f"Length of train dataloader: {len(dataloader_test)} batches of {BATCH_SIZE}")
    print(f"Length of test dataloader: {len(dataloader_train)} batches of {BATCH_SIZE}")

    img = dataset_test.__getitem__(3)

    print(f"l'immagine e di tipo{type(img)} e dimensione{img.shape}")
    # Check out what's inside the training dataloader
    train_features_batch, train_labels_batch = next(iter(dataloader_train))
    train_features_batch.shape, train_labels_batch.shape

    #model_0 = resnet50(weights=ResNet50_Weights.DEFAULT)
    #model_0 = UNET(in_channels=1, out_channels=1)


    # Set the seed and start the timer
    # torch.manual_seed(42)
    # train_time_start = timer()
    #
    # loss_fn = nn.CrossEntropyLoss() # this is also called "criterion"/"cost function" in some places
    # optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.01)
    #
    # # Set the number of epochs (we'll keep this small for faster training times)
    # epochs = 10
    #
    # # Create training and testing loop
    # for epoch in tqdm(range(epochs)):
    #     print(f"Epoch: {epoch}\n-------")
    #     ### Training
    #     train_loss = 0
    #     # Add a loop to loop through training batches
    #     for batch, (X, y) in enumerate(dataloader_train):
    #         model_0.train()
    #         # 1. Forward pass
    #         y_pred = model_0(X)
    #
    #         # 2. Calculate loss (per batch)
    #         loss = loss_fn(y_pred, y)
    #         train_loss += loss  # accumulatively add up the loss per epoch
    #
    #         # 3. Optimizer zero grad
    #         optimizer.zero_grad()
    #
    #         # 4. Loss backward
    #         loss.backward()
    #
    #         # 5. Optimizer step
    #         optimizer.step()
    #
    #         # Print out how many samples have been seen
    #         if batch % 400 == 0:
    #             print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")
    #
    #     # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    #     train_loss /= len(train_dataloader)
    #
    #     ### Testing
    #     # Setup variables for accumulatively adding up loss and accuracy
    #     test_loss, test_acc = 0, 0
    #     model_0.eval()
    #     with torch.inference_mode():
    #         for X, y in test_dataloader:
    #             # 1. Forward pass
    #             test_pred = model_0(X)
    #
    #             # 2. Calculate loss (accumatively)
    #             test_loss += loss_fn(test_pred, y)  # accumulatively add up the loss per epoch
    #
    #             # 3. Calculate accuracy (preds need to be same as y_true)
    #             test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
    #
    #         # Calculations on test metrics need to happen inside torch.inference_mode()
    #         # Divide total test loss by length of test dataloader (per batch)
    #         test_loss /= len(test_dataloader)
    #
    #         # Divide total accuracy by length of test dataloader (per batch)
    #         test_acc /= len(test_dataloader)
    #
    #     ## Print out what's happening
    #     print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")
    #
    # # Calculate training time
    # train_time_end_on_cpu = timer()
    # total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
    #                                             end=train_time_end_on_cpu,
    #                                             device=str(next(model_0.parameters()).device))
    # data.dataloader()
    # files = dm.list_files(basepath, "*")
    # files = dm.load_dict("img_and_annot")
    # img_list = []
    # label_list = []



    #
    #
    #
    # data = data.DataLoader(
    #     basepath,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     collate_fn=None,
    #     pin_memory=True,
    #)


    # for filename in files["r_img"]:
    #     img_list.append(dm.read_img(filename))
    # for filename in files["r_label"]:
    #     label_list.append(dm.read_img(filename))
    #
    #
    # print(len(label_list))
    # print(label_list[0].shape)
    # print(type(label_list[0]))
    # dv.see_random_slice(label_list[0].dataobj)
"""
import data_manipulation as dm

if __name__ == "__main__":

    data_list = dm.load_dict("img_and_annot")

    dm.convert_img(data_list["r_label"])
    dm.convert_img(data_list["r_label"])

    basepath = "/media/neuropsycad/disk12t/VascoDiogo/OASIS/FS7/"

    all_files = dm.list_files(basepath)
    dm.write_dict(all_files, "img_and_annot")
    
    basepath = "/media/neuropsycad/disk12t/VascoDiogo/OASIS/FS7/"

    all_files = list_files(basepath)

    write_file(all_files,"directory")
    #run(runner_id=None, meta_file=None, config_file=None, logging_file=None, args_file=None)
    #run(None, meta_file="configs/metadata.json", config_file="configs/train.json", logging_file="configs/logging.conf", args_file=None)



"""
