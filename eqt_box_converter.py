import py360convert
import cv2
import time

class EQT_Box_Converter():
    '''
    convert 360 EQT frame and 6-face mapping 
    py360convert Docs: https://github.com/sunset1995/py360convert
    '''

    def __init__(self):
        self.face_w = 640 #切割後每一個 cube face 的寬度
        self.eqt_w = 3840 #原始 EQT 寬度
        self.eqt_h = 1920 #原始 EQT 高度
    

    
    def setOriginalEQTSize(self, w, h):
        self.eqt_w = w
        self.eqt_h  = h
        # print(f"set size {w}, {h}")


    def slice(self, frame):
        '''
        將 EQT 切成 6 個 cube-map crop
        '''

        # record original EQT size
        self.setOriginalEQTSize(frame.shape[1], frame.shape[0])

        # py360convert convert EQT to Cube-map
        cubeMap = py360convert.e2c(frame, face_w = self.face_w)

        # slice cubemap
        # crop_img = img[y:y+h, x:x+w]
        fw = self.face_w # face width
        crop_front = cubeMap[fw:fw+fw, fw:fw*2]
        crop_right = cubeMap[fw:fw+fw, fw*2:fw*3]
        crop_back = cubeMap[fw:fw+fw, fw*3:fw*4]
        crop_left = cubeMap[fw:fw+fw, 0:fw]
        crop_up = cubeMap[0:fw, fw*1:fw*2]
        crop_down = cubeMap[fw*2:fw*3, fw*1:fw*2]
        
        # sequence: f, r, b, l, u, d
        crop_outputs = [crop_front, crop_right, crop_back, crop_left, crop_up, crop_down]

        # save crop images
        output_path = './output/'
        for i, crop in enumerate(crop_outputs):
            cv2.imwrite(f'{output_path}crop-box-{i}.jpg', crop)


        return crop_outputs
    
    
    def assemble(self, crop_frames):
        '''
        reassemble 6 cube-map crop to EQT
        py360convert restriction: a dict with 6 elements with keys 'F', 'R', 'B', 'L', 'U', 'D' each of which is a numpy array in shape of 256 x 256.
        '''
        
        # Validation
        if len(crop_frames) != 6: 
            raise ValueError(f"crop_frames len should be 6, but only get {len(crop_frames)}")

        # resize
        for i in range(len(crop_frames)):
            crop_frames[i] = cv2.resize(crop_frames[i], (255, 255))

        # formatting crop_dict
        crop_dict = {'F': None, 'R': None, 'B': None, 'L': None, 'U': None, 'D': None}
        crop_dict['F'] = crop_frames[0]
        crop_dict['R'] = cv2.flip(crop_frames[1], 1) # 1 表示水平翻轉
        crop_dict['B'] = cv2.flip(crop_frames[2], 1) # 1 表示水平翻轉
        crop_dict['L'] = crop_frames[3]
        crop_dict['U'] = crop_frames[4]
        crop_dict['D'] = crop_frames[5]

        eqt = py360convert.c2e(crop_dict, h= self.eqt_h, w=self.eqt_w, cube_format='dict')

        return eqt