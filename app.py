from huggingface_hub import InferenceClient
import gradio as gr

client = InferenceClient(
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
)


def format_prompt(message, history):
  prompt = "<s>"
  for user_prompt, bot_response in history:
    prompt += f"[INST] {user_prompt} [/INST]"
    prompt += f" {bot_response}</s> "
  prompt += f"[INST] {message} [/INST]"
  return prompt

def generate(
    prompt, history, system_prompt, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(f"{system_prompt}, {prompt}", history)
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output


additional_inputs=[
    gr.Textbox(
        label="System Prompt",
        max_lines=1,
        interactive=True,
    ),
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=256,
        minimum=0,
        maximum=1048,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penalize repeated tokens",
    )
]

examples=[["I'm planning a vacation to Japan. Can you suggest a one-week itinerary including must-visit places and local cuisines to try?", None, None, None, None, None, ],
          ["Can you write a short story about a time-traveling detective who solves historical mysteries?", None, None, None, None, None,],
          ["I'm trying to learn French. Can you provide some common phrases that would be useful for a beginner, along with their pronunciations?", None, None, None, None, None,],
          ["I have chicken, rice, and bell peppers in my kitchen. Can you suggest an easy recipe I can make with these ingredients?", None, None, None, None, None,],
          ["Can you explain how the QuickSort algorithm works and provide a Python implementation?", None, None, None, None, None,],
          ["What are some unique features of Rust that make it stand out compared to other systems programming languages like C++?", None, None, None, None, None,],
         ]

gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    additional_inputs=additional_inputs,
    title="Mixtral 46.7B",
    examples=examples,
    concurrency_limit=20,
).launch(show_api=False)