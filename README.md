# backlight-enhancer
SNU 2020-2 Computer Vision Project: Backlight Enhancer 

by Hanbin Koo, Eundo Lee, Hyunmin Choi

## Summary
The main script (`backlight_enhancement.py`) takes a single backlit image as input and enhances it using one of three methods: 
1. Histogram specification
2. Log transformation
3. Pyramid fusion

## Install requirements
```bash
pip install -r requirements.txt
```

## Run
```bash
python backlight_enhancer.py \
    --method ${METHOD_CHOICE} \
    --input-path ${PATH_TO_INPUT_IMAGE} \
    --output-dir ${PATH_TO_DIRECTORY_TO_WRITE_OUTPUT_IMAGE_TO}
```
### Arguments
#### method
`method` is a string with one of the following values:
1. `log-transformation`
2. `histogram-specification`
3. `pyramid-fusion`
#### input-path
`input-path` is a string that denotes the path to the input image.
#### output-dir
`output-dir` is a string that denotes the path to the output DIRECTORY.
