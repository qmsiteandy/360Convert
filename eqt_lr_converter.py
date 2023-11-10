import cv2

class EQT_LR_Converter():
    '''
    Convert 360 EQT frame to left right frame and 
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


        # slice cubemap
        # crop_img = img[y:y+h, x:x+w]
        crop_left = frame[0:self.eqt_h , 0:int(self.eqt_w/2)]
        crop_right = frame[0:self.eqt_h, int(self.eqt_w/2):self.eqt_w]
        
        # sequence: left, right
        crop_outputs = [crop_left, crop_right]

        # save crop images
        output_path = './output/'
        for i, crop in enumerate(crop_outputs):
            cv2.imwrite(f'{output_path}crop-lr-{i}.jpg', crop)

        return crop_outputs
    
    
    def assemble(self, crop_frames):
        '''
        reassemble left & right crop to EQT
        '''
        
        # Validation
        if len(crop_frames) != 2: 
            raise ValueError(f"crop_frames len should be 2, but only get {len(crop_frames)}")

        eqt = cv2.hconcat(crop_frames)

        return eqt