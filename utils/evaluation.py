import open_clip
import torch
import PIL
import json
import os
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(filename = "eval.log", level = logging.INFO)

def get_eval_dataset(path):
    with open(path, 'r') as file:
        eval_dataset = json.load(file)
        eval_dataset = eval_dataset['scenes']

    return eval_dataset


def get_eval_model():
    # Load the model and tokenizer
    model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')

    return model, preprocess, tokenizer

def eval_different_dataset(eval_dataset1, eval_dataset2, preprocess, model, tokenizer, device, logging):
    # This evaluation pipeline is used to evaluate the video quality
    cosine = torch.nn.functional.cosine_similarity
    print('Evaluating caption and image similarity on sampled original and refined datasets')
    total_score1 = 0
    total_score2 = 0

    # Evaluate the similarity between the caption and the image of randomized and refined datasets
    # Both of these datasets have been recaptioned
    for i, data in enumerate(zip(eval_dataset1, eval_dataset2)):
        data1, data2 = data
        # Load the image, caption and recaption
        frames_path1, caption1 = data1['frames_path'], data1['recaption']
        frames_path2, caption2 = data2['frames_path'], data2['recaption']
        
        text_input1 = tokenizer([caption1])
        text_input2 = tokenizer([caption2])
        text_input1 = text_input1.to(device)
        text_input2 = text_input2.to(device)

        frame_list1 = os.listdir(frames_path1)
        frame_list2 = os.listdir(frames_path2)

        with torch.inference_mode():
            text_features1 = model.encode_text(text_input1)
            text_features2 = model.encode_text(text_input2)

        sim1, sim2 = 0, 0

        length = min(len(frame_list1), len(frame_list2))

        for i in range(0, length, 2):
            endpath1 = frame_list1[i]
            endpath2 = frame_list2[i]
            imagepath1 = frames_path1 + '/' + endpath1
            imagepath2 = frames_path2 + '/' + endpath2

            image1 = PIL.Image.open(imagepath1)
            image2 = PIL.Image.open(imagepath2)

            image_input1 = preprocess(image1).unsqueeze(0)
            image_input2 = preprocess(image2).unsqueeze(0)

            image_input1 = image_input1.to(device)
            image_input2 = image_input2.to(device)

            with torch.inference_mode():
                image_features1 = model.encode_image(image_input1)
                image_features2 = model.encode_image(image_input2)
        
            # Calculate the similarity between the image and the caption of these two datasets
            sim1 += max(100 * cosine(image_features1, text_features1), 0)
            sim2 += max(100 * cosine(image_features2, text_features2), 0)
        
        sim1 = sim1.item() / length
        sim2 = sim2.item() / length

        # Update the total score of each dataset
        total_score1 += sim1
        total_score2 += sim2 
        print(f'Random Dataset similarity: {total_score1/(i+1)} | Filtered Dataset similarity: {total_score2/(i+1)}')
        logging.info(f'Random Dataset similarity: {total_score1/(i+1)} | Filtered Dataset similarity: {total_score2/(i+1)}')

    total_score1 = total_score1 / len(eval_dataset1)
    total_score2 = total_score2 / len(eval_dataset2)
    
    return total_score1, total_score2 
    

def eval_same_dataset(eval_dataset, preprocess, model, tokenizer, device, logging):
    # This evaluation pipeline is used to evaluate the caption quality
    cosine = torch.nn.functional.cosine_similarity
    print('Evaluating caption and recaption similarity on the same dataset')
    total_caption_score = 0
    total_recaption_score = 0

    for i, data in enumerate(eval_dataset):
        # Load the image, caption and recaption
        frames_path, caption, recaption = data['frames_path'], data['caption'], data['recaption']

        caption_input = tokenizer([caption])
        recaption_input = tokenizer([recaption])
        caption_input = caption_input.to(device)
        recaption_input = recaption_input.to(device)
    
        caption_image_sim = 0
        recaption_image_sim = 0

        with torch.inference_mode():
            caption_features = model.encode_text(caption_input)
            recaption_features = model.encode_text(recaption_input)

        frame_list = os.listdir(frames_path)
        for i in range(0, len(frame_list), 2):
            endpath = frame_list[i]
            imagepath = frames_path + '/' + endpath
            image = PIL.Image.open(imagepath)
            image_input = preprocess(image).unsqueeze(0)
            image_input = image_input.to(device)
            with torch.inference_mode():
                image_features = model.encode_image(image_input)

            # Calculate the similarity between the image and the caption
            caption_image_sim += max(100 * cosine(image_features, caption_features), 0)
            recaption_image_sim += max(100 * cosine(image_features, recaption_features), 0) 

        caption_score = caption_image_sim.item() / len(frame_list)
        recaption_score = recaption_image_sim.item() / len(frame_list)
        
        # Update the total score of each dataset
        total_caption_score += caption_score
        total_recaption_score += recaption_score
        print(f'image {i+1} | caption: {total_caption_score/(i+1)} | recaption: {total_recaption_score/(i+1)}')
        logging.info(f'image {i+1} | caption: {total_caption_score/(i+1)} | recaption: {total_recaption_score/(i+1)}')
    
    total_caption_score = total_caption_score / len(eval_dataset)
    total_recaption_score = total_recaption_score / len(eval_dataset)
    return total_caption_score, total_recaption_score


if __name__ == "__main__":
    model, preprocess, tokenizer = get_eval_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using device: {device}')
    model = model.to(device)

    # Evaluation if caption and recaption similarity to image on different datasets
    eval_dataset1 = get_eval_dataset('./frame_score_results/random_scenes_100-109.json')
    eval_dataset2 = get_eval_dataset('./frame_score_results/filtered_scenes_100-109.json')

    total_score1, total_score2 = eval_different_dataset(eval_dataset1, eval_dataset2, preprocess, model, tokenizer, device, logging)
    
    # Evaluation if caption and recaption similarity to image on the same dataset
    eval_dataset = get_eval_dataset('./frame_score_results/filtered_scenes_100-109.json')
    
    caption_score, recaption_score = eval_same_dataset(eval_dataset, preprocess, model, tokenizer, device, logging)

    






