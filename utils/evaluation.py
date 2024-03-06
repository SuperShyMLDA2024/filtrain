import open_clip
import time
import torch
import PIL

def get_eval_data():
    """
    Please implement this later on!
    """
    return None

def get_eval_model():
    model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')

    return model, preprocess, tokenizer

def eval_different_dataset(eval_dataset1, eval_dataset2, preprocess, model, tokenizer, device):
    cosine = torch.nn.functional.cosine_similarity
    print('Evaluating caption and image similarity on sampled original and refined datasets')
    total_score1 = 0
    total_score2 = 0


    for i, data in enumerate(zip(eval_dataset1.values(), eval_dataset2.values())):
        data1, data2 = data
        image1, caption1 = data1['image'], data1['recaption']
        image2, caption2 = data2['image'], data2['recaption']
            
        image1 = PIL.Image.open(image1)
        image2 = PIL.Image.open(image2)

        image_input1 = preprocess(image1).unsqueeze(0)
        image_input2 = preprocess(image2).unsqueeze(0)
            
        text_input1 = tokenizer([caption1])
        text_input2 = tokenizer([caption2])

        with torch.inference_mode():
            image_features1 = model.encode_image(image_input1)
            image_features2 = model.encode_image(image_input2)
            text_features1 = model.encode_text(text_input1)
            text_features2 = model.encode_text(text_input2)

        sim1 = cosine(image_features1, text_features1)
        sim2 = cosine(image_features2, text_features2)
            
        total_score1 += sim1.item()
        total_score2 += sim2.item() 
        print(f'Dataset 1 similarity: {sim1.item()/(i+1)} | Dataset 2 similarity: {sim2.item()/(i+1)}')

    total_score1 = total_score1 / len(eval_dataset1)
    total_score2 = total_score2 / len(eval_dataset2)
        
    return total_score1, total_score2 
    

def eval_same_dataset(eval_dataset, preprocess, model, tokenizer, device):
    cosine = torch.nn.functional.cosine_similarity
    print('Evaluating caption and recaption similarity on the same dataset')
    total_caption_score = 0
    total_recaption_score = 0

    for i, data in enumerate(eval_dataset.values()):
        
        image, caption, recaption = data['image'], data['caption'], data['recaption']
        image = PIL.Image.open(image)

        image_input = preprocess(image).unsqueeze(0)
        caption_input = tokenizer([caption])
        recaption_input = tokenizer([recaption])
        
        with torch.inference_mode():
            image_features = model.encode_image(image_input)
            caption_features = model.encode_text(caption_input)
            recaption_features = model.encode_text(recaption_input)

        caption_image_sim = cosine(image_features, caption_features)
        recaption_image_sim = cosine(image_features, recaption_features)
        
        total_caption_score += caption_image_sim.item()
        total_recaption_score += recaption_image_sim.item()
        print(f'image {i+1} | caption: {caption_image_sim.item()/(i+1)} | recaption: {recaption_image_sim.item()/(i+1)}')
    
    total_caption_score = total_caption_score / len(eval_dataset)
    total_recaption_score = total_recaption_score / len(eval_dataset)
    return total_caption_score, total_recaption_score


if __name__ == "__main__":
    model, preprocess, tokenizer = get_eval_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using device: {device}')

    eval_dataset1 = get_eval_data()
    eval_dataset2 = get_eval_data()
    eval_dataset = get_eval_data()

    total_score1, total_score2 = eval_different_dataset(eval_dataset1, eval_dataset2, preprocess, model, tokenizer, device)
    caption_score, recaption_score = eval_same_dataset(eval_dataset, preprocess, model, tokenizer, device)

    






