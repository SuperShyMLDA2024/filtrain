## Installation Setup 
To make the conda env, use:
```conda env create -f environment.yml```

After creating the conda, activate it using:
```conda activate mlda```

Run the python file:
```python utils/inference.py```

## Notes
Don't forget to run it from root folder (".dummy-1")
Use ```python utils/get_first_n_videos.py``` to get the first few videos for easier testing.

## Dataset
Video -> Clips -> Scene Split
No Video: 18750
No Clips: 560755
No Scenes: 1504248