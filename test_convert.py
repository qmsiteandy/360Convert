import time
import cv2
# from eqt_box_converter import EQT_Box_Converter 
# from eqt_lr_converter import EQT_LR_Converter 
from eqt_crop_converter import EQT_Crop_Converter 
# from dewarp_eqt_converter import Dewarp_EQT_Converter

input_file = "frame_20.jpg"
input_file_path = './source/'
output_path = './output/'



if __name__ == "__main__":

    eqtConverter = EQT_Crop_Converter()

    frame = cv2.imread(input_file_path+input_file)

    timer = time.time()

    crop_outputs = eqtConverter.slice(frame, slice_count=3, slice_shift=300)


    print(f"slice cost time: {time.time() - timer}")
    timer = time.time()
    
    assembled_eqt = eqtConverter.assemble(crop_outputs)

    print(f"assemble cost time: {time.time() - timer}")
    timer = time.time()

    output_path = './output/'
    cv2.imwrite(f'{output_path}assemble.jpg', assembled_eqt)

   