class DummyImageProcessor:
    def __init__(self, *args, **kwargs):
        self.dummy = True
        pass

    def preprocess(self, image, *args, **kwargs):
        return image

    def postprocess(self, image, *args, **kwargs):
        return image
