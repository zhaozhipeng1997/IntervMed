import torch
from transformers import AutoProcessor
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvision import transforms
import gc

def Mydebias(model,LAYER_IDX):
    IDX = int(LAYER_IDX.split(":")[-1])
    if "l" in LAYER_IDX:
        model.language_model.model.layers[IDX].mlp.gate_proj.weight.data *= 0.7
        model.language_model.model.layers[IDX].mlp.up_proj.weight.data *= 0.7
    elif "v" in LAYER_IDX:
        model.vision_tower.vision_model.encoder.layers[IDX].mlp.fc1.weight.data *= 0.7
        model.vision_tower.vision_model.encoder.layers[IDX].mlp.fc2.weight.data *= 0.7

    return model
def debias_layer(layer, alpha=0.4, num_components=32):
    with torch.no_grad():
        weight = layer.weight.data
        original_dtype = weight.dtype
        if original_dtype == torch.float16:
            weight = weight.float()
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        
        for i in range(min(num_components, len(S))):
            S[i] = S[i] * (1 - alpha * (1 - i/num_components)) 
        
        reconstructed_weight = U @ torch.diag(S) @ Vh
        if original_dtype == torch.float16:
            reconstructed_weight = reconstructed_weight.half()
        
        layer.weight.data = reconstructed_weight

def analyze_layer_impact(processor, model, device, bias_img, bias_txt, unbias_img, unbias_txt,
                         target_layer="multi_modal_projector", max_new_tokens=20, intervention_strength=0.8):
    # llava1.5
    # bias_inputs = processor(
    #     text=f"USER: <image>\n{bias_txt}\nASSISTANT:",
    #     images=bias_img,
    #     return_tensors="pt",
    #     padding=True,
    #     truncation=True
    # ).to(device)


    # unbias_inputs = processor(
    #     text=f"USER: <image>\n{unbias_txt}\nASSISTANT:",
    #     images=unbias_img,
    #     return_tensors="pt",
    #     padding=True,
    #     truncation=True
    # ).to(device)

    # llava-next
    conversation_bias = [
            {
              "role": "user",
              "content": [
                  {"type": "text", "text": bias_txt},
                  {"type": "image"},
                ],
            },
        ]
    prompt_bias = processor.apply_chat_template(conversation_bias, add_generation_prompt=True)
    bias_inputs = processor(images=bias_img, text=prompt_bias, return_tensors="pt").to("cuda:0")
    conversation_unbias = [
            {
              "role": "user",
              "content": [
                  {"type": "text", "text": unbias_txt},
                  {"type": "image"},
                ],
            },
        ]
    prompt_unbias = processor.apply_chat_template(conversation_unbias, add_generation_prompt=True)
    unbias_inputs = processor(images=unbias_img, text=prompt_unbias, return_tensors="pt").to("cuda:0")


    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        bias_outputs = model.generate(
            **bias_inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True
        )
    bias_prompt_length = bias_inputs['input_ids'].shape[1]
    bias_prob, bias_response = get_yes_probability_and_response(bias_outputs, processor, bias_prompt_length)


    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        unbias_outputs = model.generate(
            **unbias_inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True
        )
    unbias_prompt_length = unbias_inputs['input_ids'].shape[1]
    unbias_prob, unbias_response = get_yes_probability_and_response(unbias_outputs, processor, unbias_prompt_length)


    ate = bias_prob - unbias_prob


    debug = False
    if debug:
        print(f"\n[Debug] Intervention Intensity: {intervention_strength}")
        print(f"[Debug] bias_prob: {bias_prob:.6f}, unbias_prob: {unbias_prob:.6f}, ATE: {ate:.6f}")


    if intervention_strength == 0.0:
        if debug:
            print(f"[Debug] Intervention strength is 0, skipping intervention process, AIE=0")
        return ate, 0.0


    ref_activation = collect_activation(model, unbias_inputs, target_layer)
    if ref_activation is None:
        print(f"Warning: Unable to collect activation values ​​for layer {target_layer}.")
        return ate, 0.0


    intervened_logits = perform_intervention(model, bias_inputs, ref_activation, target_layer, intervention_strength)
    if intervened_logits is None:
        return ate, 0.0


    intervened_prob, _ = get_yes_probability_from_logits(
        intervened_logits, bias_inputs['input_ids'].shape[1], processor
    )


    aie = bias_prob - intervened_prob

    if debug:
        print(f"[Debug] intervened_prob: {intervened_prob:.6f}, AIE: {aie:.6f}")
        print(f"[Debug] Contribution Rate: {aie/ate*100 if ate != 0 else 0:.2f}%")
    if intervention_strength > 0.9 and abs(intervened_prob - unbias_prob) > 0.1:
        print(f"Warning: High intervention strength ({intervention_strength}) results in a large difference between intervened_prob ({intervened_prob:.4f}) and unbias_prob ({unbias_prob:.4f})")

    return ate, aie



def get_yes_probability_and_response(outputs, processor, prompt_length=None):
    """Retrieve the probability of 'yes' and the complete response (calculated using logits)."""

    if hasattr(outputs, 'sequences'):
        
        full_response = processor.decode(outputs.sequences[0], skip_special_tokens=True)
        sequences = outputs.sequences
    elif isinstance(outputs, dict) and 'sequences' in outputs:

        full_response = processor.decode(outputs['sequences'][0], skip_special_tokens=True)
        sequences = outputs['sequences']
    else:
        full_response = outputs
        sequences = None

    assistant_part = full_response.split("ASSISTANT:")[-1].strip()
    if prompt_length is not None and hasattr(outputs, 'scores') and outputs.scores is not None:
        scores_list = outputs.scores
        if len(scores_list) > 0:
            first_gen_logits = scores_list[0]  # (batch_size, vocab_size)
            probs = torch.softmax(first_gen_logits, dim=-1)
            target_id = processor.tokenizer.convert_tokens_to_ids("yes")
            if target_id is not None:
                prob = probs[0, target_id].item()  # batch_size=1
                if prob < 0.3 and any(word in assistant_part.lower() for word in ["yes", "yeah", "yep"]):
                    prob = min(prob + 0.5, 1.0)
                return prob, assistant_part

    return calculate_yes_probability_from_text(assistant_part), assistant_part

def calculate_yes_probability_from_text(response):
    yes_tokens = ["yes", "yeah", "yep", "affirmative", "certainly", "definitely"]

    response_lower = response.lower()

    # 
    if any(token in response_lower for token in yes_tokens):
        return 0.8
    elif any(token in response_lower for token in ["no", "not", "none", "nothing"]):
        return 0.2
    else:
        return 0.5

def get_yes_probability_from_logits(logits, prompt_length, processor):
    response = processor.decode(logits[0].argmax(dim=-1), skip_special_tokens=True)
    
    prob = calculate_probability_from_logits(logits, prompt_length, processor, "yes")
    
    if prob < 0.3 and any(word in response.lower() for word in ["yes", "yeah", "yep"]):
        return min(prob + 0.5, 1.0), response
    
    return prob, response

def calculate_probability_from_logits(logits, prompt_length, processor, target_token):
    if logits.shape[1] > prompt_length:
        first_gen_logits = logits[0, prompt_length, :]
    else:
        first_gen_logits = logits[0, -1, :]
    
    probs = torch.softmax(first_gen_logits, dim=-1)
    target_id = processor.tokenizer.convert_tokens_to_ids(target_token)
    
    if target_id is not None:
        return probs[target_id].item()
    else:
        return 0.0

def collect_activation(model, inputs, layer_spec):
    activation = [None]
    
    layer = get_layer(model, layer_spec)
    if layer is None:
        return None
    
    def collection_hook(module, input, output):
        if isinstance(output, tuple):
            activation[0] = output[0].detach().clone()
        else:
            activation[0] = output.detach().clone()
    
    handle = layer.register_forward_hook(collection_hook)
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        model(**inputs)
    
    handle.remove()
    
    return activation[0]

def perform_intervention(model, inputs, ref_activation, layer_spec, intervention_strength=0.8):
    layer = get_layer(model, layer_spec)
    if layer is None:
        return None

    intervention_strength = max(0.0, min(1.0, intervention_strength))

    if intervention_strength == 0.0:
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**inputs)
        return outputs.logits

    def intervention_hook(module, input, output):
        if isinstance(output, tuple):
            mixed_activation = intervention_strength * ref_activation + (1 - intervention_strength) * output[0]
            return (mixed_activation,) + output[1:]
        else:
            return intervention_strength * ref_activation + (1 - intervention_strength) * output

    handle = layer.register_forward_hook(intervention_hook)

    try:
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**inputs)
        return outputs.logits
    finally:
        handle.remove()

def get_layer(model, layer_spec):
    try:
        if layer_spec == "multi_modal_projector":
            return model.multi_modal_projector
        elif layer_spec.startswith("v:"):
            layer_index = int(layer_spec.split(":")[1])
            return model.vision_tower.vision_model.encoder.layers[layer_index]
        elif layer_spec.startswith("l:"):
            layer_index = int(layer_spec.split(":")[1])
            return model.language_model.model.layers[layer_index]
        else:
            parts = layer_spec.split('.')
            current = model
            for part in parts:
                current = getattr(current, part, None)
                if current is None:
                    break
            return current
    except (AttributeError, IndexError) as e:
        print(f"Error: Layer {layer_spec} not found - {str(e)}")
        return None
