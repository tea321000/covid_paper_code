from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunet.paths import nnUNet_raw_data
import pandas as pd

if __name__ == '__main__':
    task_name="Task200_COVID_19_20"
    # 定义训练和测试文件夹
    downloaded_data_dir = "/dataset/377293fc/v1"
    patients_csv = join(downloaded_data_dir, 'COVID-19-20_TrainValidation.xlsx')
    downloaded_trained_dir = join(downloaded_data_dir, 'Train')
    downloaded_val_dir = join(downloaded_data_dir, 'Validation')

    trained_patients = pd.read_excel(patients_csv, sheet_name='Train set', usecols=['FILENAME'])
    val_patients = pd.read_excel(patients_csv, sheet_name='Validation set', usecols=['FILENAME'])

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesVal = join(target_base, "imagesVal")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)


    for p in trained_patients['FILENAME']:
        shutil.copy(join(downloaded_trained_dir, p + "_ct.nii.gz"), join(target_imagesTr,  p + "_0000.nii.gz"))
        shutil.copy(join(downloaded_trained_dir, p + "_seg.nii.gz"), join(target_labelsTr,  p + ".nii.gz"))

    for p in val_patients['FILENAME']:
        shutil.copy(join(downloaded_val_dir, p + "_ct.nii.gz"), join(target_imagesVal,  p + "_0000.nii.gz"))


    json_dict = {}
    json_dict['name'] = "COVID_19_20_50%"
    json_dict['description'] = "lung segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "COVID data for nnunet"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "lesion",
    }
    #
    json_dict['numTraining'] = len(trained_patients)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             trained_patients['FILENAME']]
    json_dict['test'] = []
    save_json(json_dict, os.path.join(target_base, "dataset.json"))