from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, os, re
import pandas as pd
#torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

model_name = "ahmed-masry/unichart-chartqa-960"
images_paths=[]
path1="../images_data/AD-Average/"
path2="../images_data/AD-features/"
path3="../images_data/AD-PCA/"
path4="../images_data/AD-sampling/"


file_names=[]
prompts=[]
responses=[]

png_files=[]
for filename in os.listdir(path1):
    if filename.endswith('.png'):
        png_files.append(path1+filename)


for filename in os.listdir(path2):
    if filename.endswith('.png'):
        png_files.append(path2+filename)

for filename in os.listdir(path3):
    if filename.endswith('.png'):
        png_files.append(path3+filename)

for filename in os.listdir(path4):
    if filename.endswith('.png'):
        png_files.append(path4+filename)

for image_path in png_files:


    input_prompts=[
        "<opencqa> Explain the Graph <s_answer>",
        "<opencqa> Explain the Chart <s_answer>",
        "<summarize_chart> <s_answer>",
        
    ]
    
    #image_path = "./content/chart_example_1.png"
    #input_prompt = "<summarize_chart> Give Summary <s_answer>"
    for input_prompt in input_prompts:
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        processor = DonutProcessor.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        image = Image.open(image_path).convert("RGB")
        decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = processor(image, return_tensors="pt").pixel_values

        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=4,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        sequence = processor.batch_decode(outputs.sequences)[0]
        #print(image_path)
        path_segments=image_path.split("/")
        #print(path_segments[2]+path_segments[3])
        #print(sequence)
        file_names.append(path_segments[2]+path_segments[3])
        prompts.append(input_prompt)
        responses.append(sequence)
        #sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        #print(sequence)
        #sequence = sequence.split("<s_answer>")[1].strip()
        #print(sequence)

data=[file_names,prompts,responses]
df = pd.DataFrame(data)
df.to_excel("UniChart_responses.xlsx", index=False) 