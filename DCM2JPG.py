# pip install aspose-imaging-python-net

import os
from tqdm import tqdm
from aspose.imaging import Image
from aspose.imaging.imageoptions import JpegOptions

dicom_extension = '.dcm'
root_folder = 'D:\\University\\LuanVan\\KiTS_dataset' # Đường dẫn đến folder chứa file DICOM

total_dicom_files = sum(
    1 for _, _, filenames in os.walk(root_folder) for filename in filenames if filename.lower().endswith(dicom_extension)
)
print('Tổng số file DICOM trong thư mục: ', total_dicom_files)

def convert_DCM2JPG(INPUT, OUTPUT):      
    # Tạo và thiết lập đối tượng của lớp ImageOptionsBase, ở đây là JpegOptions
    jpeg_options = JpegOptions()
            
    # Gọi phương thức image.Save, lưu file với đường dẫn và phần mở rộng JPG
    try:
        with Image.load(INPUT) as image:
            image.save(OUTPUT, jpeg_options)
            return 1
    except RuntimeError as e:
        txt_file_path = dicom_file_path[:-4] + '_error.txt'
        f = open(txt_file_path, 'w')
        f.write(f'{e}')
        f.close()
        return 0

if __name__ == "__main__":
    for dirpath, dirnames, filenames in os.walk(root_folder):
        total_dicom_files_in_small_folder = sum(
            1 for filename in filenames if filename.lower().endswith(dicom_extension)
        )
        for filename in tqdm(filenames, total=total_dicom_files_in_small_folder, desc='Đang thực hiện chuyển đổi'):
            if filename.lower().endswith(dicom_extension):
                dicom_file_path = os.path.join(dirpath, filename) # Load file DICOM
                jpg_file_path = dicom_file_path[:-4] + '.jpg' # Lưu file với đường dẫn và phần mở rộng JPG
                
                if convert_DCM2JPG(dicom_file_path, jpg_file_path):
                    os.remove(dicom_file_path) # Xóa file DICOM nếu chuyển đổi thành công