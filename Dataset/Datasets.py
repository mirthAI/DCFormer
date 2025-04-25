import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm
import matplotlib.pyplot as plt

def plot(im, cmap=None):
    plt.figure()
    plt.imshow(im, cmap=cmap)
def visualize(images, x, y, cmap='gray'):
    h, w, _ = images.shape
    a = np.zeros((int(h * x), int(w * y)))
    k = 0
    m = 0
    for i in range(x):
        l = 0
        for j in range(y):
            a[k:k + h, l:l + w] = images[..., m]
            l += w
            m += 1

        k += h
    plot(a, cmap=cmap)
    

class DATASET:
    def __init__(self, 
                 args,
                 debug=False,
                 subs=True,
                 mae=False,
                train_metadata='metadata/train_metadata.csv',
                val_metadata="metadata/validation_metadata.csv",
                non_path_train='nones.csv',
                non_path_val='non_nones_val.csv',
                reports_file_train= "radiology_text_reports/train_reports.csv",
                reports_file_valid= "radiology_text_reports/validation_reports.csv",
                train_data_folder= "data_volumes/dataset/train",
                val_data_folder = "data_volumes/dataset/valid",
                labels = "valid_labels.csv",
                target_shape=(480, 480, 240),
                finetune=False,
                retrieval=False,
                ) -> None:
        if finetune:
            self.train_dataset = CTReportDatasetinfer(args, data_folder=train_data_folder, csv_file=reports_file_train, metadata=train_metadata, non_path=non_path_train, labels =args.data_path + 'train_labels.csv', target_shape=target_shape, subs=subs, mae=mae, debug=debug)
        else:
            self.train_dataset = CTReportDataset(args, data_folder=train_data_folder, csv_file=reports_file_train, metadata=train_metadata, non_path=non_path_train, target_shape=target_shape, debug=debug)

        if retrieval:
            self.test_dataset = CTReportDataset(args, data_folder=val_data_folder, csv_file=reports_file_valid, metadata=val_metadata, non_path=non_path_train, target_shape=target_shape, debug=debug)
        else:
            self.test_dataset = CTReportDatasetinfer(args, data_folder=val_data_folder, csv_file=reports_file_valid, metadata=val_metadata, non_path=non_path_val, labels = labels, target_shape=target_shape, subs=subs, mae=mae, debug=debug)

    

class CTReportDataset(Dataset):
    def __init__(self, args, data_folder, csv_file, metadata, non_path, target_shape=(480,480,240), min_slices=20, resize_dim=500, debug=False):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.target_shape = target_shape
        self.df = pd.read_csv(metadata) #select the metadata
        self.check_df = list(pd.read_csv(non_path)['VolumeName']) #select the metadata

        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()[:args.batch_size * args.world_size] if debug else self.prepare_samples()
        percent = 100
        #num_files = int((len(self.samples) * percent) / 100)
        #self.samples = self.samples[:num_files]
        print('Train data size: ', len(self.samples))



        #self.resize_dim = resize_dim
        #self.resize_transform = transforms.Resize((resize_dim, resize_dim))
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text


    def prepare_samples(self):
        samples = []
        for patient_folder in glob.glob(os.path.join(self.data_folder, '*')):
        # for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                for nii_file in glob.glob(os.path.join(accession_folder, '*')):

                # for nii_file in glob.glob(os.path.join(accession_folder, '*.npz')):
                    if os.path.basename(nii_file) not in self.check_df:
                        accession_number = nii_file.split("/")[-1]
                        # accession_number = accession_number.replace(".npz", ".nii.gz")
                        if accession_number not in self.accession_to_text:
                            continue

                        impression_text = self.accession_to_text[accession_number]

                        if impression_text == "Not given.":
                            impression_text=""

                        input_text_concat = ""
                        for text in impression_text:
                            input_text_concat = input_text_concat + str(text)
                        input_text = f'{impression_text}'
                        samples.append((nii_file, input_text_concat))
                        self.paths.append(nii_file)
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        #nii_img = nib.load(str(path))
        img_data = self.read_nii_data(path)
        file_name = os.path.basename(path)

        row = self.df[self.df['VolumeName'] == file_name]
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])

        # Define the target spacing values
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)


        img_data = slope * img_data + intercept
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = ((img_data / 1000)).astype(np.float32)

        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        resized_array = self.resize_array(tensor, current, target)
        resized_array = resized_array[0][0]


        img_data= np.transpose(resized_array, (1, 2, 0))
        img_data = img_data*1000
        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data+400 ) / 600)).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = self.target_shape
        
        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)
        
        tensor = tensor.unsqueeze(0)

        return tensor
    
    def read_nii_data(self, file_path):
        """
        Read NIfTI file data.

        Args:
        file_path (str): Path to the NIfTI file.

        Returns:
        np.ndarray: NIfTI file data.
        """
        try:
            nii_img = nib.load(file_path)
            nii_data = nii_img.get_fdata()
            return nii_data
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def resize_array(self, array, current_spacing, target_spacing):
        """
        Resize the array to match the target spacing.

        Args:
        array (torch.Tensor): Input array to be resized.
        current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
        target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

        Returns:
        np.ndarray: Resized array.
        """
        # Calculate new dimensions
        original_shape = array.shape[2:]
        scaling_factors = [
            current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
        ]
        new_shape = [
            int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
        ]
        # Resize the array
        resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
        return resized_array    
    
    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')  

        # return nii_file, video_tensor, input_text
        return video_tensor, input_text


class CTReportDatasetinfer(Dataset):
    def __init__(self, args, data_folder, csv_file, metadata, non_path=None, subs=True, target_shape=(480,480,240), min_slices=20, resize_dim=500, force_num_frames=True, labels = "labels.csv", debug=False, mae=False):
        self.mae = mae
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.target_shape = target_shape
        self.df = pd.read_csv(metadata) #select the metadata
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()[:args.world_size * 16] if debug else self.prepare_samples()
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)
        print('Test data size: ', len(self.samples))

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text


    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        for patient_folder in patient_folders:
        # for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]                        
                    # accession_number = accession_number.replace(".npz", ".nii.gz")
                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]
                    text_final = ""
                    for text in list(impression_text):
                        text = str(text)
                        if text == "Not given.":
                            text = ""

                        text_final = text_final + text

                    onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
                    if len(onehotlabels) > 0:
                        if nii_file.split('/')[-1] not in ['train_1267_a_4.nii.gz', 'train_11755_a_3.nii.gz', 'train_11755_a_4.nii.gz']:

                            samples.append((nii_file, text_final, onehotlabels[0]))
                            self.paths.append(nii_file)
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        
        img_data = self.read_nii_data(path)

        file_name = os.path.basename(path)

        row = self.df[self.df['VolumeName'] == file_name]
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])

        # Define the target spacing values
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)

        img_data = slope * img_data + intercept
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = ((img_data / 1000)).astype(np.float32)

        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        resized_array = self.resize_array(tensor, current, target, path)
        resized_array = resized_array[0][0]

        img_data= np.transpose(resized_array, (1, 2, 0))
        img_data = img_data*1000
        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data+400 ) / 600)).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = self.target_shape
        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)


        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        return tensor
    def read_nii_data(self, file_path):
        """
        Read NIfTI file data.

        Args:
        file_path (str): Path to the NIfTI file.

        Returns:
        np.ndarray: NIfTI file data.
        """
        try:
            nii_img = nib.load(file_path)
            nii_data = nii_img.get_fdata()
            return nii_data
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def resize_array(self, array, current_spacing, target_spacing, path=None):
        """
        Resize the array to match the target spacing.

        Args:
        array (torch.Tensor): Input array to be resized.
        current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
        target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

        Returns:
        np.ndarray: Resized array.
        """
        # Calculate new dimensions
        original_shape = array.shape[2:]
        scaling_factors = [
            current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
        ]
        try:
            new_shape = [
                int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
        except Exception as e:
            print(f"Error in path: {path}, Exception: {e}")

        resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()



        
        # Resize the array
        return resized_array
    
    def __getitem__(self, index):
        nii_file, input_text, onehotlabels = self.samples[index]
        # print(nii_file)
        video_tensor = self.nii_to_tensor(nii_file)
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')  
        name_acc = nii_file.split("/")[-2]
        if self.mae:
            return video_tensor, input_text
        return nii_file, video_tensor, onehotlabels
        # return video_tensor, input_text, onehotlabels, name_acc
