import gradio as gr
import torch
from PIL import Image
from transformers import CLIPProcessor
from safetensors.torch import load_file
import os

from clip import CLIP
from text_encoder import TextEncoder, PositionalEmbedding
from vision_encoder import VisionEncoder
from text_decoder import TextDecoder
from image_captioning_model import ImageCaptioningModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_PATH = "clip_atasoglu/flickr8k-dataset_model/model.safetensors"
CAPTIONING_MODEL_PATH = (
    "clip_atasoglu/flickr8k-dataset_model/captioning/model.safetensors"
)

def load_models_and_processor(clip_path,captioning_path):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    text_vocab_size = processor.tokenizer.vocab_size

    vision_encoder = VisionEncoder(
        d_model=768,
        img_size=224,
        patch_size=16,
        n_channels=3,
        n_heads=12,
        n_layers=12,
        emb_dim=512,
    )

    text_encoder = TextEncoder(
        vocab_size=text_vocab_size,
        d_model=512,
        max_seq_len=77,
        n_layers=6,
        n_heads=8,
        emb_dim=512, 
    )
    
    text_decoder = TextDecoder(
        vocab_size=text_vocab_size,
        max_seq_len=77,
        n_layers=6,
        n_heads=8,
        embed_dim=512, 
        hidden_dim=768,
    )
    
    clip_model = CLIP(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        temperature=0.07,
        embed_dim=512,
        device="cuda"
    )
    
    state_dict = load_file(clip_path)
    
    clip_model.load_state_dict(state_dict, strict=False)
    clip_model.to(DEVICE)
    clip_model.eval()
    
    for params in clip_model.parameters():
        params.requires_grad = False
        
    captioning_model = ImageCaptioningModel(
        clip_model=clip_model,
        text_decoder=text_decoder,
    )

    cap_state_dict = load_file(captioning_path)
    captioning_model.load_state_dict(cap_state_dict, strict=False)
    captioning_model.to(DEVICE)
    captioning_model.eval()
    
    return processor,clip_model,captioning_model

processor, clip_model, captioning_model = load_models_and_processor(
    CLIP_MODEL_PATH,
    CAPTIONING_MODEL_PATH
)

@torch.inference_mode()  
def generate_caption(image_input):
    if image_input is None:
        return "Please upload an image."
    try:
        print("--- Preparing image for captioning ---")
        inputs = processor(images=image_input, return_tensors="pt").to(DEVICE)

        print("--- Generating caption IDs ---")
        outputs = captioning_model(
            pixel_values=inputs["pixel_values"]
        )  
        generated_ids = outputs["generated_ids"]
        print(f"Generated IDs shape: {generated_ids.shape}")
        print(f"Generated IDs tensor: {generated_ids}")

        print("--- Decoding IDs ---")
        
        decoded_batch = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(f"Output of batch_decode: {decoded_batch}")
        print(f"Type of batch_decode output: {type(decoded_batch)}")

        
        if isinstance(decoded_batch, list) and len(decoded_batch) > 0:
            
            generated_text = decoded_batch[0]
            print(
                f"Selected text item: '{generated_text}' (type: {type(generated_text)})"
            )
            
            if isinstance(generated_text, str):
                return generated_text.strip()
            else:
                print("ERROR: Decoded item is not a string!")
                return f"Error: Decoded item type is {type(generated_text)}"
        elif isinstance(decoded_batch, list) and len(decoded_batch) == 0:
            print("ERROR: batch_decode returned an empty list.")
            return "Error: Could not decode caption (empty result)."
        else:
            print(
                f"ERROR: Unexpected output type from batch_decode: {type(decoded_batch)}"
            )
            
            try:
                return str(decoded_batch).strip()
            except:
                return "Error: Could not decode caption (unexpected format)."
        

    except Exception as e:
        print(f"Captioning Exception: {e}")
        import traceback

        traceback.print_exc()  
        return f"Error generating caption: {e}"




@torch.inference_mode() 
def classify_image(image_input, labels_text):
    
    labels = []
    try:
        if image_input is None:
            return "Please upload an image."
        if not labels_text:
            return "Please enter comma-separated labels."

        print("--- Parsing labels ---")
        
        labels = [label.strip() for label in labels_text.split(',') if label.strip()]
        print(f"Parsed labels: {labels}")
        if not labels:
            return "No valid labels entered."
        prompts = [f"a photo of a {label}" for label in labels]
        print(f"Generated prompts: {prompts}")

        print("--- Processing Image ---")
        image_inputs = processor(images=image_input, return_tensors="pt").to(DEVICE)
        image_features = clip_model.encode_image(image_inputs['pixel_values'])
        image_features /= image_features.norm(dim=-1, keepdim=True) 
        print(f"Image features shape: {image_features.shape}")

        print("--- Processing Text Prompts ---")
        text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(DEVICE)
        text_features = clip_model.encode_text(text_inputs['input_ids'], text_inputs['attention_mask'])
        text_features /= text_features.norm(dim=-1, keepdim=True) 
        print(f"Text features shape: {text_features.shape}")


        print("--- Calculating Similarity ---")
        
        logit_scale = clip_model.logit_scale.exp()
        print(f"Logit scale: {logit_scale.item()}")
        
        similarity = (image_features @ text_features.T) * logit_scale
        print(f"Similarity shape: {similarity.shape}")
        probs = similarity.softmax(dim=-1)
        print(f"Probs shape: {probs.shape}")

        
        probs_squeezed = probs.squeeze()
        print(f"Squeezed Probs shape: {probs_squeezed.shape}")
        print(f"Number of labels: {len(labels)}")

        
        if probs_squeezed.dim() != 1 or probs_squeezed.shape[0] != len(labels):
             error_msg = (f"Shape mismatch! Labels count: {len(labels)}, "
                          f"Probabilities count: {probs_squeezed.shape[0]}")
             print(f"ERROR: {error_msg}")
             return f"Error: {error_msg}"
        

        print("--- Formatting results ---")
        
        results = {label: prob.item() for label, prob in zip(labels, probs_squeezed)}
        print(f"Final Results: {results}")
        return results

    except Exception as e:
        
        print(f"Classification Exception Type: {type(e)}")
        print(f"Classification Exception Args: {e.args}")
        import traceback
        traceback.print_exc() 
        
        return f"Error during classification: {e}"



with gr.Blocks() as demo:
    gr.Markdown("CLIP Model and Image Captioning Demo")
    gr.Markdown("Models trained on Flickr8k dataset")
    
    with gr.Tabs():
        with gr.TabItem("Image Captioining"):
            with gr.Row():
                caption_image_input = gr.Image(type="pil",label="Upload Image")
                caption_output = gr.Textbox(label="Generated Caption",interactive=False)
            caption_button = gr.Button("Generate Caption")
            
        
        with gr.TabItem("Zero-Shot Image Classification"):
            with gr.Row():
                classify_image_input = gr.Image(type="pil", label="Upload Image")
                with gr.Column():
                    classify_labels_input = gr.Textbox(label="Enter Comma-Separated Labels", placeholder="e.g., dog, cat, person, car")
                    classify_output = gr.Label(num_top_classes=5, label="Classification Results")
            classify_button = gr.Button("Classify Image")
    
    caption_button.click(fn=generate_caption, inputs=caption_image_input, outputs=caption_output)
    classify_button.click(fn=classify_image, inputs=[classify_image_input, classify_labels_input], outputs=classify_output)

if __name__ == "__main__":
    demo.launch(share=False)
