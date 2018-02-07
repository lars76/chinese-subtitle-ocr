# chinese-subtitle-ocr
This project shows how to implement an optical character recognition via SSD and CNN. The focus lies here mainly on the special case of Chinese subtitles. Due to their homogeneous form, a near perfect accuracy of 100% should be possible (see for example [1]). The method used in this project is susceptible to changing lightning conditions and the resolution of the video, but by fine-tuning the parameters for most videos a good accuracy can be achieved.

The recognition of the subtitles is done in three steps:

1. Detection of the subtitle region in the frame using the *SSD: Single Shot MultiBox Detector* [2].

2. Finding the position of the characters by filtering the image and applying an adaptive threshold.

3. Recognition of the individual characters via a convolutional neural network.

## 1. SSD
Normally in videos the position of the subtitles will stay the same. This is why only once at the start of the program a bounding box has to be found. To make sure that the right box was detected, we will iterate over several different frames.

The *Tensorflow Object Detection API* with its SSD implementation is used internally to find the box [3]. As pretrained model "ssd_mobilenet_v1_coco" was chosen. To generate the data set, 423 video frames were annotated manually and split into 338 training samples and 85 validation samples.

It's also possible to generate automatically the training data by adding random text to images. In the tools/detection folder a python script for doing this can be found.

Training was done for 200K steps and took about 10 hours, at 163K steps the loss was at its lowest point.

![Total loss](https://i.imgur.com/z7fmydY.png)

## 2. Filtering and adaptive threshold
On the region that contains the subtitle two filters are applied: first a bilateral filter because it preserves edges, then a Gaussian blur to remove the remaining noise. Finally, an adaptive threshold produces the following image:

![Before and after thresholding](https://i.imgur.com/MaX9g4g.png)

Now to find the regions we reduce the dimensionality of our image. All the pixels in our image are summed column-wise from left to right. By applying this transformation we lose information like shape and form but spaces between characters can be easily detected. Furthermore, we know the width of a character, because it's approximately as wide as the height of our detected subtitle region.

![Detected regions](https://i.imgur.com/eVZNjg1.png)

After having grouped the regions, the real width and the distance between characters can be determined. Because we consider multiple frames, these character attributes are to +-2 pixels accurate.

## 3. OCR
Lastly the image patches are passed on to our CNN. The network architecture was not optimized but rather just chosen to be similar to that of a VGG convnet with very few layers [4].

For training 40K random background images were extracted from 5 videos. For each Chinese character, 300 images (240 for training, 60 for validation) were chosen from the 32x32 random patches. The 14 different *Google Noto Fonts* were used to draw the characters.

Due to not having enough computing power, only 804 Chinese characters (803 + 1 space) were generated. Without a GPU each epoch took about 1h. In total 192960 training images and 48240 validation images had to be processed. Training was stopped at 96% validation accuracy.

![Final result](https://i.imgur.com/i8uoIjC.jpg)

## Discussion
The general method of this project seems to be working fairly well, but a lot of small changes have to be made to achieve better results. To see this, let's analyze some of the errors that were made by the program.

![Errors](https://i.imgur.com/skKb6F5.png)

Here "拨", "角色" wasn't recognized because it wasn't among our 804 Chinese characters. "三" is in the dictionary but our training data didn't have enough  variations (more shifts, size changes etc. are needed).

![Errors 2](https://i.imgur.com/Xh79l68.png)

Additionally, there are some errors introduced by the region detection. The black line at the end confuses the program. This can be solved by filtering more, using an overlap when the columns are summed or by setting a threshold for the black pixels.

In the end it might be better to skip the filtering part and use directly another neural network that performs the detection and the optical character recognition.

## References
[1] Y. Xu, S. Shan, Z. Qiu, Z. Jia, Z. Shen, Y. Wang, M. Shi, and E. I.-C. Chang, “End-to-end subtitle detection and recognition for videos in East Asian languages via CNN ensemble,” Signal Processing: Image Communication, vol. 60, pp. 131–143, Feb. 2018.

[2] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, “SSD: Single Shot MultiBox Detector,” Lecture Notes in Computer Science, pp. 21–37, 2016.

[3] https://github.com/tensorflow/models/tree/master/research/object_detection

[4] K. Simonyan, A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” CoRR, 2014.