import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging
import traceback
from PIL import Image

from load_config import get_config
from image_processing import pil_to_base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_reply(model, processor, img, question):
    try:
        # Ensure img is a PIL Image object
        if isinstance(img, str):
            img = Image.open(img)
        
        # Convert image to base64
        img_base64 = pil_to_base64(img)
        img_uri = 'data:image;base64,' + img_base64
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_uri},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info - handle possible boolean return
        vision_info = process_vision_info(messages)
        if isinstance(vision_info, bool):
            raise ValueError(f"process_vision_info returned a boolean: {vision_info}")
        
        image_inputs, video_inputs = vision_info

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to the same device as model
        device = model.device
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        
        # Handle batch processing
        generated_ids_trimmed = []
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
            # Convert tensors to lists for slicing
            in_ids_list = in_ids.tolist()
            out_ids_list = out_ids.tolist()
            generated_ids_trimmed.append(out_ids_list[len(in_ids_list):])
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Return string instead of list
        if isinstance(output_text, list) and len(output_text) > 0:
            return output_text[0]
        return str(output_text)
        
    except Exception as e:
        logger.error(f"Error in get_reply: {e}")
        logger.error(traceback.format_exc())
        return f"Error processing request: {str(e)}"

if __name__=='__main__':
    config=get_config()
    model_name=config['model_name']
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(model_name,use_fast=True)

    img_path='../pics/image_0001.jpg'
    question=input('Ask a question based on this image: ').strip()
    reply = get_reply(model, processor, img_path, question)
    print(reply)
