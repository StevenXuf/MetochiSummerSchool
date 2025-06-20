from PIL import Image
import io
import base64

def pil_to_base64(pil_img: Image.Image, form: str = "png") -> str:
    buffered = io.BytesIO()
    pil_img.save(buffered, format=form)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


if __name__=='__main__':
    pil_to_base64(Image.new('RGB',(480,640),"rgb(255,0,255)"))
