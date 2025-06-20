import os
import torch
import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import pipeline
from IPython.display import Audio

from chatbot import get_reply
from load_config import get_config

def main():
    config=get_config()
    model_name=config['model_name']
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    tts_output=pipeline("text-to-speech", model="suno/bark-small")
    processor = AutoProcessor.from_pretrained(model_name,use_fast=True)

    def inference(image,question):
        answer=get_reply(model,processor,image,question)
        tts_output = tts_pipeline(answer)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with open(temp_file.name, "wb") as f:
            f.write(tts_output["audio"])
        return answer,temp_file.name

    interface = gr.Interface(
        fn=inference,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Textbox(lines=2, placeholder="Ask a question about the image...", label="Question")
        ],
        outputs=[gr.Textbox(label="Answer"),
                gr.Audio(label="Audio Answer", autoplay=True)],
        title="Vision-Language Q&A",
        description="Upload an image and ask a question. The model will try to answer it!"
    )

    interface.queue()
    interface.launch()

if __name__=='__main__':
    main()
