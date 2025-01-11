import random


class RandomRotationCustom:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return img.rotate(angle)


class SyncTransform:
    def __init__(self, transform):
        self.transform = transform
        self.random_state = None

    def __call__(self, img):
        if self.random_state is None:
        
            self.random_state = random.getstate()

        random.setstate(self.random_state)
        transformed_img = self.transform(img)

        return transformed_img