

import random
import cv2
import numpy as np


class RandomHSV(object):
    def __init__(self,
                 hue_delta: int = 5,
                 saturation_delta: int = 30,
                 value_delta: int = 30,
                 prob: int = 0.5
                 ) -> None:
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta
        self.prob = prob
    
    def _get_hsv_gains(self):
        hsv_gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta, self.saturation_delta, self.value_delta
        ]
        # random selection of h, s, v
        hsv_gains *= np.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(np.int16)
        return hsv_gains

    def __call__(self, img) -> dict:
        
        if random.uniform(0,1) > self.prob:
            return img
        
        hsv_gains = self._get_hsv_gains()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        # cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)
        img = cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR)

        return img
