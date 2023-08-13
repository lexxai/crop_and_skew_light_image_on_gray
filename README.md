# crop_and_skew_light_image_on_gray

Using Python and OpenCV to detect the border of a light image on a gray background, crops and corrects its geometry.

## help

```
ai_crop_images   [-h] (--images IMAGES | --image IMAGE) [--output OUTPUT] [--gamma GAMMA] [--morph MORPH] [--normalize NORMALIZE] [--dilate]
                  [--ratio RATIO] [--min_height MIN_HEIGHT] [--detection_height DETECTION_HEIGHT] [--debug] [--noskip] [--all_input] [-V]

options:
  -h, --help            show this help message and exit
  --images IMAGES       Directory of images to be scanned
  --image IMAGE         Path to single image to be scanned
  --output OUTPUT       Directory to output result images, default: 'output'
  --gamma GAMMA         Gamma image correction pre-filter, default: '4.0', 1 - Off
  --morph MORPH         morph image correction for smooth contours, default: '35'. 0 - Off
  --normalize NORMALIZE
                        normalize_scale image correction pre-filter, default: '1'. 1 - Off, 1.2 - for start
  --dilate              dilate, CV operation to close open contours with an eclipse
  --ratio RATIO         desired correction of the image aspect ratio H to W, default: '1.294'
  --min_height MIN_HEIGHT
                        desired minimum height of the output image in px, default: '1000'
  --detection_height DETECTION_HEIGHT
                        internally downscale the original image to this height in px for the found border, default: '900'
  --debug               debug, CV operation for single image only
  --noskip              no skip wrong images, like output same size, or result less than 800x1000. Copy original if problem. Default: skipped
  --all_input           Scan all images in the input folder without skipping the search for already processed images in the output folder
  -V, --version         show version

```

### Output

```
ai_crop_images --images input
total input files: 21, ready for operations: 18

 *** Start:  2023-08-11 06:53:33.518012
File: '1889-1 copy 10.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:33.877357, duration: 0:00:00.359345

 *** Start:  2023-08-11 06:53:33.877357
File: '1889-1 copy 11.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:34.234034, duration: 0:00:00.356677

 *** Start:  2023-08-11 06:53:34.234034
File: '1889-1 copy 12.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:34.623524, duration: 0:00:00.389490

 *** Start:  2023-08-11 06:53:34.626529
File: '1889-1 copy 13.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:34.975117, duration: 0:00:00.348588

 *** Start:  2023-08-11 06:53:34.975117
File: '1889-1 copy 14.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:35.515769, duration: 0:00:00.540652

 *** Start:  2023-08-11 06:53:35.562646
File: '1889-1 copy 15.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:35.956760, duration: 0:00:00.394114

 *** Start:  2023-08-11 06:53:35.958761
File: '1889-1 copy 16.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:36.429999, duration: 0:00:00.471238

 *** Start:  2023-08-11 06:53:36.429999
File: '1889-1 copy 17.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:36.844286, duration: 0:00:00.414287

 *** Start:  2023-08-11 06:53:36.846288
File: '1889-1 copy 18.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:37.228736, duration: 0:00:00.382448

 *** Start:  2023-08-11 06:53:37.231739
File: '1889-1 copy 19.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:37.608624, duration: 0:00:00.376885

 *** Start:  2023-08-11 06:53:37.611018
File: '1889-1 copy 20.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:37.975368, duration: 0:00:00.364350

 *** Start:  2023-08-11 06:53:37.977308
File: '1889-1 copy 5.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:38.352394, duration: 0:00:00.375086

 *** Start:  2023-08-11 06:53:38.355935
File: '1889-1 copy 6.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:38.746188, duration: 0:00:00.390253

 *** Start:  2023-08-11 06:53:38.749189
File: '1889-1 copy 7.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:39.110797, duration: 0:00:00.361608

 *** Start:  2023-08-11 06:53:39.114015
File: '1889-1 copy 8.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:39.490485, duration: 0:00:00.376470

 *** Start:  2023-08-11 06:53:39.493498
File: '1889-1 copy 9.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:39.883549, duration: 0:00:00.390051

 *** Start:  2023-08-11 06:53:39.899195
File: '1889-1 copy.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:40.319622, duration: 0:00:00.420427

 *** Start:  2023-08-11 06:53:40.321625
File: '1889-1.jpg' size=1368128 bytes, modified='08/08/23 00:28:21'
Original image dimension: 3024 x 4032
Result   image dimension: 1072 x 1387
 *** End:  2023-08-11 06:53:41.263624, duration: 0:00:00.941999
100% (18 of 18) |###########################################################################| Elapsed Time: 0:00:07 Time:  0:00:07
2023-08-11 06:53:41.263624

```
