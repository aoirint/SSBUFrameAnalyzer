import pyocr
import pyocr.builders as builders
from PIL import Image

class SSBUNameRecognizer:
    def __init__(self):
        tools = pyocr.get_available_tools()
        tool = tools[0]

        langs = tool.get_available_languages()

        self.tool = tool
        self.lang = 'jpn'

        builder = builders.TextBuilder(tesseract_layout=6)
        self.builder = builder

    def __call__(self, img): # GRAY npimg
        tool = self.tool
        lang = self.lang
        builder = self.builder

        pimg = Image.fromarray(img, mode='L')

        text = tool.image_to_string(pimg, lang=lang, builder=builder)
        return text
