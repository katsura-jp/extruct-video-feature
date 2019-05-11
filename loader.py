import numpy as np
from fastprogress import progress_bar, master_bar
import cv2
import torch

class VideoLoader:
    def __init__(self, root_dir, n_clip=16, discard=True, to_tensor=True, overlap=0.0, resolution=(112, 112)):
        self.root_dir = root_dir
        self.n_clip = n_clip
        self.discard = discard
        self.to_tensor = to_tensor
        self.channel = 3
        self.resolution = resolution
        self.overlap = overlap

        # calculate overlap frames
        self._next = int(self.n_clip - self.n_clip*self.overlap)

        self._i = 0
        self._end_flag = True

        self._start_frames = None

        self._frame = None
        self._file_name = None
        self._time_depth = None
        self._frame_height = None
        self._frame_width = None
        self._video_data = None

        self.mb = master_bar(self.root_dir)
        self.mb_iter = iter(self.mb)

    def __iter__(self):
        return self

    def __next__(self):
        # if self._i == len(self.root_dir) and self._end_flag:
        #     raise StopIteration()

        if self._end_flag:
            # initialize video
            try:
                video_file = next(self.mb_iter)
            except:
                raise StopIteration()

            # video_file = self.root_dir[self._i]  # video path (e.g. /mnt/data1/dataset/Charades/Charades_v1/001YG.mp4)
            self._file_name = video_file.split('/')[-1].split('.')[0]  # video file name (e.g. 001YG)
            self._end_flag = False  # if capture is end, it is True.
            self._frame = 0  # memorize frame

            # input video data
            capture = cv2.VideoCapture(video_file)  # load video data
            self._time_depth = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self._frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

            self._video_data = data = np.zeros((self._time_depth, self.resolution[0], self.resolution[1], self.channel),
                                               dtype=np.float32)

            for count in progress_bar(range(self._time_depth), parent=self.mb):
                retaining, frame = capture.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
                frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]))
                if not retaining:
                    break
                if frame is None:
                    print('frame is None')
                else:
                    frame = frame.astype(np.float32)
                self._video_data[count, :, :, :] = frame
            self.mb.write('{} : load {}'.format(self._i, self._file_name))
            self._start_frames = []
            self._frame = 0

            for f in range(0, self._time_depth - self.n_clip, self._next):
                self._start_frames.append(f)

            # inc self._i(index number)
            self._i += 1

        if self.discard:
            if self._frame < len(self._start_frames):
                name = self._file_name + '_{:.1f}_{:.1f}'.format(self._start_frames[self._frame] + 1,
                                                                 self._start_frames[self._frame] + 1 + self.n_clip)  # e.g. 001YG_1.0_17.0
                unit = self._video_data[self._start_frames[self._frame]: self._start_frames[self._frame] + self.n_clip, :, :, :]
                if self.to_tensor:
                    unit = self.toTensor(unit)

                self._frame += 1
                return name, unit
            else:
                self._end_flag = True
                return self.__next__()
        else:
            raise NotImplementedError()

    def toTensor(self, frames):
        # frame is np.array (timedepth, height, width, channel)
        tensor = torch.from_numpy(frames)
        tensor = tensor.permute((3, 0, 1, 2))  # TxHxWxC -> CxTxHxW
        return tensor