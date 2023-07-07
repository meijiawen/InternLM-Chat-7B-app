import gradio as gr
from gradio.themes.utils import colors, fonts, sizes
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer#, pipeline, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import time
import numpy as np
from torch.nn import functional as F
import os
from threading import Thread

print(f"Starting to load the model to memory")
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b",trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True, torch_dtype=torch.float16)

if torch.cuda.is_available():
    model = model.cuda()
print(f"Sucessfully loaded the model to the memory")

model = model.eval()

def chat(message, history):
    print(history)
    with torch.no_grad():
        try:
            msg, history = model.chat(tokenizer, message, [])
        except:
            return "", history
        return "", history

class OpenGVLab(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )


gvlabtheme = OpenGVLab(primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        )

with gr.Blocks(theme=gvlabtheme) as demo:
    # history = gr.State([])
    gr.Markdown('<h1 align="center"><a href="https://github.com/InternLM/InternLM"><img src="https://raw.githubusercontent.com/InternLM/InternLM/main/doc/imgs/logo.svg" alt="InternLM-Chat" border="0" style="margin: 0 auto" /></a> </h1>')
    gr.HTML('''Due to VRAM limitations, the examples of Gradio do not include context.''')
    chatbot = gr.Chatbot().style(height=500)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(label="Chat Message Box", placeholder="Hi~ Introduce yourself!",
                             show_label=False).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
    #system_msg = gr.Textbox(
    #    start_message, label="System Message", interactive=False, visible=False)

    submit_event = msg.submit(fn=chat, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=True)
    submit_click_event = submit.click(fn=chat, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=True)
    
    stop.click(fn=None, inputs=None, outputs=None, cancels=[
               submit_event, submit_click_event], queue=False)
    clear.click(lambda: None, None, [chatbot], queue=False)

demo.queue(max_size=32, concurrency_count=2)
demo.launch()
