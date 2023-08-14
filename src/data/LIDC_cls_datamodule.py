import os 
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms 
from typing import Any, Dict, Optional, Tuple
import time
import csv
import pandas as pd

class LIDC_cls_Dataset(Dataset):
    def __init__(self, nodule_path, mode, meta_path,
                  img_size=[64, 64], degrees=180):

        # nodule_path: path to dataset nodule image folder
        # clean_path: path to dataset clean image folder
        super().__init__()   
        self.nodule_path = nodule_path
        self.meta_path = meta_path
        self.mode = mode
        self.num_classes = 1
        self.transform = transforms.Compose([transforms.Resize(img_size,antialias=True),
                                             transforms.RandomRotation(degrees)])
        # define function to get list of (image, mask)
        self.file_list = self._get_file_list()

    def __len__(self):
        return len(self.file_list)
    
    def _get_file_list(self):
        file_list = []
        metadata = pd.read_csv(self.meta_path)

        for image_path in self.nodule_path:
            
            image = np.load(image_path)            
            image = torch.from_numpy(image).to(torch.float)
            image = image.unsqueeze(0)

            img_name = image_path.split('/')[-1][:-4]
            malignancy = metadata[metadata['original_image']==img_name]['malignancy'].iloc[0]
            malignancy = 1 if malignancy >=3 else 0
            malignancy = torch.from_numpy(np.array([malignancy]))
            # malignancy = torch.nn.functional.one_hot(malignancy, num_classes=self.num_classes).to(torch.float32)
            
            file_list.append((image, malignancy))
        print(f"Len {self.mode}: {len(file_list)}")
        return file_list

    def __getitem__(self, index):
        image, malignancy = self.file_list[index]
        return self.transform(image),malignancy
    

class LIDC_cls_DataModule(LightningDataModule):
    def __init__(
        self,
        nodule_dir,
        meta_path,
        train_val_test_split: Tuple[int, int, int] = (3, 1, 1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_nodule: int = 1000,
        img_size=[128, 128],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.nodule_dir = nodule_dir
        self.meta_path = meta_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory


        # get all file_name in folder

        file_nodule_list = []
        self.num_nodule = num_nodule

        # get full path of each nodule file
        for root, _, files in os.walk(self.nodule_dir):
            for file in files:
                if file.endswith(".npy"):
                    dicom_path = os.path.join(root, file)
                    file_nodule_list.append(dicom_path)
        

        file_nodule_list = file_nodule_list[:self.num_nodule]


        nodule_train, nodule_val, nodule_test = self.split_data(file_nodule_list, train_val_test_split)

        self.data_train = LIDC_cls_Dataset(nodule_train, mode="train", meta_path=meta_path, img_size=img_size)

        self.data_val = LIDC_cls_Dataset(nodule_val, mode="valid", meta_path=meta_path, img_size=img_size)

        self.data_test = LIDC_cls_Dataset(nodule_test, mode="test", meta_path=meta_path, img_size=img_size)


    def split_data(self, file_paths, train_val_test_split):
        # get len files
        num_files = len(file_paths)
        
        # ratio
        train_ratio, val_ratio, test_ratio = train_val_test_split
        
        # get num train, val, test
        num_train = int(num_files * train_ratio / (train_ratio + val_ratio + test_ratio))
        num_val = int(num_files * val_ratio / (train_ratio + val_ratio + test_ratio))
        
        # get random index
        train_paths = list(np.random.choice(file_paths, num_train, replace=False))
        val_paths = list(np.random.choice(list(set(file_paths) - set(train_paths)), num_val, replace=False))
        test_paths = list(set(file_paths) - set(train_paths) - set(val_paths))
        return train_paths, val_paths, test_paths
        
    
    @property
    def num_classes(self):
        return 4

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

if __name__ == "__main__":
    datamodule = LIDC_cls_DataModule(num_nodule=10)
    train_dataloader = datamodule.train_dataloader()
    batch_image = next(iter(train_dataloader))
    images, labels = batch_image
    print(labels)