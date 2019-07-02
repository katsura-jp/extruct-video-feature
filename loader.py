import numpy as np
from fastprogress import progress_bar, master_bar
import cv2
import torch

class VideoLoader(object):
    def __init__(self, root_dir, n_clip=16, discard=True, to_tensor=True, overlap=0.0, resolution=(112, 112)):
        self.root_dir = root_dir # list of file paths
        self.n_clip = n_clip # clip size
        self.discard = discard # if last frames is fewer than clip size, it is discarded.
        self.to_tensor = to_tensor # For PyTorch model
        self.channel = 3 # RGB
        self.resolution = resolution # input resolution
        self.overlap = overlap # clip overlap

        # calculate overlap frames
        self._next = int(self.n_clip - self.n_clip*self.overlap) # slide size

        self._i = 0 # video file index
        self._end_flag = True # if True, loading video file

        self._start_frames = None # clip start frame

        self._frame = None # frame number
        self._file_name = None # video file name
        self._time_depth = None # video frame length
        self._frame_height = None # video frame height resolution
        self._frame_width = None # video frame width resolution
        self._video_data = None # video data array (np.array)

        self.mb = master_bar(self.root_dir) # progress bar
        self.mb_iter = iter(self.mb) # video file iterator

        # self.root_dir_iter = iter(self.root_dir)


    def __iter__(self):
        return self

    def __next__(self):
        # if self._i == len(self.root_dir) and self._end_flag:
        #     raise StopIteration()

        if self._end_flag:
            # initialize video
            try:
                video_file_path = next(self.mb_iter) # video file path
            except:
                raise StopIteration()

            # video_file = self.root_dir[self._i]  # video path (e.g. /mnt/data1/dataset/Charades/Charades_v1/001YG.mp4)
            self._file_name = video_file_path.split('/')[-1].split('.')[0]  # video file name (e.g. 001YG)
            self._end_flag = False  # if capture is end, it is True.
            self._frame = 0  # memorize frame

            # input video data
            # TODO: try-except
            try:
                capture = cv2.VideoCapture(video_file_path)  # load video data
                self._time_depth = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self._frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

                self._video_data = np.zeros((self._time_depth, self.resolution[0], self.resolution[1], self.channel),
                                                   dtype=np.float32)

                for count in progress_bar(range(self._time_depth), parent=self.mb):
                    retaining, frame = capture.read()
                    if not retaining:
                        break
                    if frame is None: # if frame is None, Zero Padding
                        self._time_depth -= 1
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
                    frame = cv2.resize(frame, (self.resolution[1], self.resolution[0])) # resize
                    frame = frame.astype(np.float32) # int --> float
                    frame /= 255 # standarize
                    #TODO: normalize?(for ImageNet or ActivityNet)
                    self._video_data[count, :, :, :] = frame
                    
                self.mb.write('{} : load {}'.format(self._i, self._file_name))
                # print('{} : load {}'.format(self._i, self._file_name))
                self._start_frames = [] # list of start frame of clip
                # self._frame = 0

                for f in range(0, self._time_depth - self.n_clip, self._next):
                    self._start_frames.append(f)

                # inc self._i(index number)
                self._i += 1
            except:
                print('error {} : {}'.format(self._i, self._file_name))
                self._end_flag = True  # if capture is end, it is True.
                self._i += 1
                return self.__next__()

        if self._frame < len(self._start_frames):
            name = self._file_name + '_{:.1f}_{:.1f}'.format(self._start_frames[self._frame] + 1,
                                                             self._start_frames[self._frame] + 1 + self.n_clip)  # e.g. 001YG_1.0_17.0
            unit = self._video_data[self._start_frames[self._frame]: self._start_frames[self._frame] + self.n_clip, :, :, :]
            if self.to_tensor:
                unit = self.toTensor(unit)

            self._frame += 1
            return name, unit
        else:
            if self.discard:
                self._end_flag = True
                return self.__next__()
            else:
                name = self._file_name + '_{:.1f}_{:.1f}'.format(self._start_frames[self._frame] + 1,
                                                                 self._time_depth)
                unit = self._video_data[self._start_frames[self._frame]: self._time_depth, :, :, :]
                if self.to_tensor:
                    unit = self.toTensor(unit)
                self._end_flag = True
                return name, unit

    def toTensor(self, frames):
        # frame is np.array (timedepth, height, width, channel)
        tensor = torch.from_numpy(frames)
        tensor = tensor.permute((3, 0, 1, 2))  # TxHxWxC -> CxTxHxW
        return tensor



def test():
    video_loader = VideoLoader(root_dir, n_clip=16, discard=True, to_tensor=True, overlap=0.0, resolution=(112, 112)):



if __name__ == '__main__':
    test()
