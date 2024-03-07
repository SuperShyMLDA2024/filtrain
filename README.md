## Installation Setup 
To make the conda env, use:
```conda env create -f environment.yml```

After creating the conda, activate it using:
```conda activate mlda```

For recaptioning, get api key from https://aistudio.google.com/app/apikey and put it inside of .env as the following:
```GEMINI_API_KEY=<YOUR_API_KEY>```

Run the python file to run filtering and recaptioning:
```python inference.py```
Use config.yaml to change the batch to run.

## Notes
Don't forget to run it from root folder (".dummy-1")
Use ```python utils/get_first_n_videos.py``` to get the first few videos for easier testing.

## Dataset
Video -> Clips -> Scene Split
No Video: 18750
No Clips: 560755
No Scenes: 1504248

## Where to find the video information JSON file
https://drive.google.com/drive/folders/1R52ueV6czG4M409YC6ca7W5QUqhCyoXg