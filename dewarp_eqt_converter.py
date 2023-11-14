import cv2
import numpy as np

class Dewarp_EQT_Converter():
    '''
    Convert 360 EQT frame to 4 pic
    '''

    def __init__(self):
        self.eqt_w = 3840 #原始 EQT 寬度
        self.eqt_h = 1920 #原始 EQT 高度
    

    
    def setOriginalEQTSize(self, w, h):
        self.eqt_w = w
        self.eqt_h  = h
        # print(f"set size {w}, {h}")


    def dewarp(self, frame):
        '''
        Dewarp EQT (from ChatGPT)
        '''

        height, width = frame.shape[:2]

        # Define the equirectangular-to-cartesian transformation
        scale_factor = 1  # Adjust this factor based on the desired output size
        output_width = int(scale_factor * width)
        output_height = int(scale_factor * height)

        # Generate the dewarped image
        dewarped_image = cv2.remap(
            frame,
            np.zeros((output_height, output_width), np.float32),
            np.zeros((output_height, output_width), np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        
        return dewarped_image
    
    
    def assemble(self, crop_frames):
        '''
        reassemble left & right crop to EQT
        '''
        
        # Validation
        if len(crop_frames) != 4: 
            raise ValueError(f"crop_frames len should be 2, but only get {len(crop_frames)}")

        # assemble crops
        eqt = cv2.hconcat(crop_frames)
        # add black border
        blank_image = np.zeros((int(self.eqt_h/4), self.eqt_w, 3), dtype=np.uint8)
        eqt = cv2.vconcat([blank_image, eqt, blank_image])

        return eqt