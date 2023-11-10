import cv2
import numpy as np

class EQT_4pic_Converter():
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


    def slice(self, frame):
        '''
        Slice EQT to left and right frame
        '''

        # record original EQT size
        self.setOriginalEQTSize(frame.shape[1], frame.shape[0])


        slice_width = int(self.eqt_w/4) # width for each slice
        image_y_range = [int(self.eqt_h/4), int(self.eqt_h/4*3)] # only y index in this range has img

        # slice cubemap
        # crop_img = img[y:y+h, x:x+w]
        crop_0 = frame[image_y_range[0]:image_y_range[1] , 0:slice_width]
        crop_1 = frame[image_y_range[0]:image_y_range[1] , slice_width:slice_width*2]
        crop_2 = frame[image_y_range[0]:image_y_range[1] , slice_width*2:slice_width*3]
        crop_3 = frame[image_y_range[0]:image_y_range[1] , slice_width*3:self.eqt_w]

        
        # sequence: left, right
        crop_outputs = [crop_0, crop_1, crop_2, crop_3]

        # save crop images
        # output_path = './output/'
        # for i, crop in enumerate(crop_outputs):
        #     cv2.imwrite(f'{output_path}crop-4pic-{i}.jpg', crop)

        return crop_outputs
    
    
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