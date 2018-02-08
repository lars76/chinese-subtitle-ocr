# chinese-subtitle-ocr
This project shows how to implement an optical character recognition via SSD and CNN. The focus lies here mainly on the special case of Chinese subtitles. Due to their homogeneous form, a near perfect accuracy of 100% should be possible (see for example [1]). The method used in this project is susceptible to changing lightning conditions and the resolution of the video, but by fine-tuning the parameters for most videos a good accuracy can be achieved.

The recognition of the subtitles is done in three steps:

1. Detection of the subtitle region in the frame using the *SSD: Single Shot MultiBox Detector* [2].

2. Finding the position of the characters by filtering the image and applying an adaptive threshold.

3. Recognition of the individual characters via a convolutional neural network.

### 1. SSD
Normally in videos the position of the subtitles will stay the same. This is why only once at the start of the program a bounding box has to be found. To make sure that the right box was detected, we will iterate over several different frames.

The *Tensorflow Object Detection API* with its SSD implementation is used internally to find the box [3]. As pretrained model "ssd_mobilenet_v1_coco" was chosen. To generate the data set, 423 video frames were annotated manually and split into 338 training samples and 85 validation samples.

It's also possible to generate automatically the training data by adding random text to images. In the tools/detection folder a python script for doing this can be found.

Training was done for 200K steps and took about 10 hours, at 163K steps the loss was at its lowest point.

![Total loss](https://i.imgur.com/z7fmydY.png)

### 2. Filtering and adaptive threshold
On the region that contains the subtitle two filters are applied: first a bilateral filter because it preserves edges, then a Gaussian blur to remove the remaining noise. Finally, an adaptive threshold produces the following image:

![Before and after thresholding](https://i.imgur.com/MaX9g4g.png)

Now to find the regions we reduce the dimensionality of our image. All the pixels in our image are summed column-wise from left to right. By applying this transformation we lose information like shape and form but spaces between characters can be easily detected. Furthermore, we know the width of a character, because it's approximately as wide as the height of our detected subtitle region.

![Detected regions](https://i.imgur.com/eVZNjg1.png)

After having grouped the regions, the preliminary width and distance between characters can be determined. By doing this for several frames, we can get about +-2 pixels close to the real distance. Sometimes due to incorrectly detected regions, the width and/or the distance can be wrong.

### 3. OCR
Lastly the image patches are passed on to our CNN. The network architecture was not optimized but rather just chosen to be similar to that of a VGG convnet with very few layers [4]. An architecture adapted to the CIFAR-10/CIFAR-100 data set might produce better results (check out [5]).

For training 40K random background images were extracted from 5 videos. For each Chinese character, 300 images (240 for training, 60 for validation) were chosen from the 32x32 random patches. The 14 different *Google Noto Fonts* were used to draw the characters.

Due to not having enough computing power, only 804 Chinese characters (803 + 1 space) were generated. Without a GPU each epoch took about 1h. In total 192960 training images and 48240 validation images had to be processed. Training was stopped at 96% validation accuracy. Running the training till convergence would have produced about 99%. However, this percentage does not represent the real accuracy, because all the images are computer-generated.

Our final result looks like this:

![Final result](https://i.imgur.com/i8uoIjC.jpg)

## Discussion
The general method of this project seems to be working fairly well, but a lot of small changes have to be made to achieve better results. To see this, let's analyze some of the errors that were made by the program.

![Errors](https://i.imgur.com/skKb6F5.png)

Here "拨" and "角色" weren't recognized because they aren't among our 804 Chinese characters. "三" is in the dictionary but our training data didn't have enough variations (more shifts, size changes etc. are needed).

![Errors 2](https://i.imgur.com/Xh79l68.png)

Additionally, there are some errors introduced by the region detection. The black line in the second image confuses the program. This can be fixed by filtering more, using an overlap when the columns are summed or by setting a threshold for the black pixels. The debug mode can help to figure out where the recognition went wrong.

The filtering part seems to be the bottleneck of our program. In the end it might be thus be better to skip it and use directly another neural network that performs the detection in addition to the optical character recognition.

## Dependencies
- Pillow
- tensorflow >= 1.4
- keras
- opencv

## Training
For both training models Chinese fonts are needed. On Linux they can be found in the folder /usr/share/fonts and can be copied simply to the directory where the data will be generated.

### SSD
There are two possibilities:

1. Get some images from the internet (for example *Open Images Dataset* using **download_images.py**) or from videos (using OpenCV). Then run the script **generate_dataset_artificially.py**.
2. Extract a few hundred video frames using **extract_images.py**. Subtitles can be optionally downloaded from YouTube to speed up the procedure. The program LabelImg can be used for the annotations. Finally, run the script **generate_dataset.py**.

In both cases the last step is to use **csv_to_tfrecord.py**. Then follow the TF Object Detection API instructions [3].

If you want to use real data, my training set can be found here: https://drive.google.com/file/d/1RhJ2B9PLYDURN3PVCW7lRFS2j1PIHfWW/view?usp=sharing

For transfer learning you can use my checkpoint: https://drive.google.com/file/d/14oMxJ7Rff9gwU_51s5PR92kR90hGEn1n/view?usp=sharing

### OCR
First we have to choose which and how many Chinese characters we want to generate. The more characters you generate, the more computing power you need for training.

There are a few different character lists on the internet:

- 通用规范汉字表 (Table of General Standard Chinese Characters) which can be found here as a PDF http://www.gov.cn/gzdt/att/att/site1/20130819/tygfhzb.pdf
- Jun Da's frequency list: http://lingua.mtsu.edu/chinese-computing/statistics/index.html

You can also create your own list. This is only a few lines of code with regular expressions:

```
from collections import Counter

text = "ABC+_)]]\n简体中文简简简体体体test"
dictionary = Counter(re.sub(r"[^\u4e00-\u9fff]", "", text)).most_common(6000))
```

Then after having assembled our dictionary, the script **generate_dataset.py** can be called, followed by **train_model.py**.

## References
[1] Y. Xu, S. Shan, Z. Qiu, Z. Jia, Z. Shen, Y. Wang, M. Shi, and E. I.-C. Chang, “End-to-end subtitle detection and recognition for videos in East Asian languages via CNN ensemble,” Signal Processing: Image Communication, vol. 60, pp. 131–143, Feb. 2018.

[2] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, “SSD: Single Shot MultiBox Detector,” Lecture Notes in Computer Science, pp. 21–37, 2016.

[3] https://github.com/tensorflow/models/tree/master/research/object_detection

[4] K. Simonyan, A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” CoRR, 2014.

[5] http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d313030