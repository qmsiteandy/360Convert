import cv2
import numpy as np

class EQT_Crop_Converter():
    '''
    Convert 360 EQT frame to several crop
    '''

    def __init__(self):
        self.eqt_w = 3840 #原始 EQT 寬度
        self.eqt_h = 1920 #原始 EQT 高度
        self.slice_count = 4
        self.slice_shift = 0
    

    
    def setOriginalEQTSize(self, w, h):
        self.eqt_w = w
        self.eqt_h  = h
        # print(f"set size {w}, {h}")

    def setCountAndShift(self, slice_count = 4, slice_shift = 0):
        self.slice_count = slice_count
        self.slice_shift = slice_shift



    def slice(self, frame, slice_count = 4, slice_shift = 0):
        '''
        Slice EQT to several crop

        Input:
        - frame: the original frame.
        - slice_count: <int> slice a frame to how many crops.
        - shift: <int> first crop start from which x-index position, and the part before <shift> position will concat into the last crop.

        Output:
        - crop_outputs: <List[frame]>
        '''

        # record original EQT size
        self.setOriginalEQTSize(frame.shape[1], frame.shape[0])

        slice_width = int(self.eqt_w/slice_count) # width for each slice

        if slice_shift > slice_width:
            slice_shift %= slice_width
        
        # record slice info
        self.setCountAndShift(slice_count, slice_shift)

        image_y_range = [int(self.eqt_h/4), int(self.eqt_h/4*3)] # only y index in this range has img
        

        crop_outputs = []

        # slice cubemap
        # crop_img = img[y:y+h, x:x+w]
        for i in range(slice_count):
            x_range = [slice_width*i + slice_shift, slice_width*(i+1) + slice_shift]
            crop_outputs.append(frame[image_y_range[0]:image_y_range[1] , x_range[0]:x_range[1]])

        if slice_shift != 0:
            crop_before_shift_position = frame[image_y_range[0]:image_y_range[1], 0:slice_shift]
            crop_outputs[-1] = cv2.hconcat([crop_outputs[-1], crop_before_shift_position])


        # crop_0 = frame[image_y_range[0]:image_y_range[1] , 0:slice_width]
        # crop_1 = frame[image_y_range[0]:image_y_range[1] , slice_width:slice_width*2]
        # crop_2 = frame[image_y_range[0]:image_y_range[1] , slice_width*2:slice_width*3]
        # crop_3 = frame[image_y_range[0]:image_y_range[1] , slice_width*3:self.eqt_w]

        
        # # sequence: left, right
        # crop_outputs = [crop_0, crop_1, crop_2, crop_3]

        # save crop images
        for i, crop in enumerate(crop_outputs):
            cv2.imwrite(f'./output/crop-4pic-{i}.jpg', crop)

        return crop_outputs
    
    
    def assemble(self, crop_frames):
        '''
        reassemble left & right crop to EQT
        '''
        
        # Validation
        if len(crop_frames) != self.slice_count: 
            raise ValueError(f"crop_frames len should be {self.slice_count}, but only get {len(crop_frames)}")

        # chunk part originally is before shift position
        last_crop = crop_frames[-1]
        print(last_crop.shape[1]-self.slice_shift, last_crop.shape[1])
        crop_before_shift_position = last_crop[0:last_crop.shape[0], last_crop.shape[1]-self.slice_shift:last_crop.shape[1]]
        crop_frames[-1] = last_crop[0:last_crop.shape[0], 0:last_crop.shape[1]-self.slice_shift]
        crop_frames.insert(0, crop_before_shift_position)

        # assemble crops
        eqt = cv2.hconcat(crop_frames)

        # add black border
        blank_image = np.zeros((int(self.eqt_h/4), self.eqt_w, 3), dtype=np.uint8)
        eqt = cv2.vconcat([blank_image, eqt, blank_image])

        return eqt