from IPython.display import Markdown
import google.generativeai as genai

import json
import textwrap
import PIL.Image
import os

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

class GeminiRecaptioning:
    def __init__(self, api_key, data):
        self.api_key = api_key
        self.data = data

        genai.configure(api_key = api_key)
        self.model = genai.GenerativeModel('gemini-pro-vision')

        self.RESOLUTION = (240, 240)


    def __generate_prompt(self, images, caption):
        res = [f"The original caption to be improved is: {caption}.", 
               "Provide additional visual details from these video frames with these following rules:",
               "All lowercase",
               "No more than 100 words",
               "Only use information available in the image and no assumptions",]
        for image in images:
            res.append(image)
        
        return res
     
    def run(self, frames_folder_path: str) -> str:
        scene_name = frames_folder_path.split('/')[-1]
        video_id = '.'.join(scene_name.split('.')[0:-1])
        clip_id = '_'.join(scene_name.split('_')[0:-1])
        scene_no =  int(scene_name.split('_')[-1])

        scene_info = list(filter(lambda x: x['clip_id'] == scene_name, self.data[video_id]['clip'][clip_id+".mp4"]['scene_split']))[0]

        folder_path = f'./frames_output/{video_id}/{scene_name}'
        count = len(os.listdir(folder_path))
        indexes = []

        if count >= 3:
            indexes = [0, count // 2, count - 1]
        elif count == 2:
            indexes = [0, count - 1]
        else:
            indexes = [0]

        

        images =  []

        for i in indexes:
            image_path = f'{folder_path}/{i:0>4}.jpg'
            image = PIL.Image.open(image_path).resize(self.RESOLUTION)
            images.append(image)

        caption = scene_info['caption']
        content = self.__generate_prompt(images, caption)

        for image in images:
            content.append(image)
        response = self.model.generate_content(content)
        return response.text
    
if __name__ == '__main__':
    # Import data from  metafiles/hdvg_0.json
    with open("metafiles/hdvg_0.json", 'r') as f:
        data = json.load(f)
    print("Data loaded")

    api_key = "YOUR_API_KEY"
    # Create an instance of the GeminiRecaptioning class
    gemini_recaptioning = GeminiRecaptioning(api_key, data)

    # Run the recaptioning process
    folder_path = './frames_output/1QFn-Yh2Y9k'

    # gets all folder in the frames_output/1QFn-Yh2Y9k
    folder_lists = list(os.listdir(folder_path))
    results = gemini_recaptioning.run_threaded([f'{folder_path}/{folder}' for folder in folder_lists])

    print(results)
       
       