import py360convert
import cv2

input_file = "frame_20.jpg"
input_file_path = './source/'
output_path = './output/'

# 输出六个面的图像文件路径
# output_faces_paths = ['front.jpg', 'back.jpg', 'top.jpg', 'bottom.jpg', 'left.jpg', 'right.jpg']

class EQT_Converter():
    '''
    convert 360 EQT frame and 6-face mapping 
    py360convert Docs: https://github.com/sunset1995/py360convert
    '''

    def __init__(self):
        self.face_w = 640 #切割後每一個 cube face 的寬度
        self.equirectangular_w = 3840 #原始 EQT 寬度
        self.equirectangular_h = 1920 #原始 EQT 高度
        pass

    
    def setOriginalEQTSize(self, w, h):
        self.equirectangular_w = w
        self.equirectangular_h  = h
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
        output_path = './slice-debug-output/'
        for i, crop in enumerate(crop_outputs):
            cv2.imwrite(f'{output_path}crop-{i}.jpg', crop)


        return crop_outputs
    
    
    def assemble(self, crop_frames):
        '''
        reassemble 6 cube-map crop to EQT
        py360convert restriction: a dict with 6 elements with keys 'F', 'R', 'B', 'L', 'U', 'D' each of which is a numpy array in shape of 256 x 256.
        '''
        
        # Validation
        if len(crop_frames) < 6: 
            raise ValueError(f"crop_frames len should be 6, but only get {len(crop_frames)}")
        
        crop_dict = {'F': None, 'R': None, 'B': None, 'L': None, 'U': None, 'D': None}
        crop_dict['F'] = cv2.resize(crop_frames[0], (256, 256))
        crop_dict['R'] = cv2.flip(cv2.resize(crop_frames[1], (256, 256)), 1) # 1 表示水平翻轉
        crop_dict['B'] = cv2.flip(cv2.resize(crop_frames[2], (256, 256)), 1) # 1 表示水平翻轉
        crop_dict['L'] = cv2.resize(crop_frames[3], (256, 256))
        crop_dict['U'] = cv2.resize(crop_frames[4], (256, 256))
        crop_dict['D'] = cv2.resize(crop_frames[5], (256, 256))


        # save crop images
        # output_path = './assemble-debug-output/'
        # for i, crop in enumerate(crop_dict.values()):
        #     cv2.imwrite(f'{output_path}crop-{i}.jpg', crop)

        eqt = py360convert.c2e(crop_dict, h= self.equirectangular_h, w=self.equirectangular_w, cube_format='dict')

        return eqt




if __name__ == "__main__":

    eqtConverter = EQT_Converter()

    frame = cv2.imread(input_file_path+input_file)
    crop_outputs = eqtConverter.slice(frame)

    assembled_eqt = eqtConverter.assemble(crop_outputs)

    output_path = './output/'
    cv2.imwrite(f'{output_path}assemble.jpg', assembled_eqt)
