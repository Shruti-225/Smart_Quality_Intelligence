# Smart Quality Intelligence
# Final Vision Experiment Report

## 1. Objective

The vision experiments investigated whether image-based foreground
segmentation and local contrast features can help identify defective
screw regions.

The experiments used the screw image dataset and its available
ground-truth defect masks. All experiments were conducted separately
from the production model.

## 2. Dataset

- Image resolution used: 128 × 128
- Defect categories:
  - manipulated_front
  - scratch_head
  - scratch_neck
  - thread_side
  - thread_top
- Defective images evaluated: 119
- Manipulated-front images evaluated in the focused analysis: 24

## 3. Foreground Segmentation Method Comparison


## 3.1 Overall Method Comparison

| Method | Images | Mean_Foreground_Percent | Mean_Defect_Inside_Percent | Mean_Defect_Outside_Percent | Std_Defect_Inside_Percent |
| --- | --- | --- | --- | --- | --- |
| otsu_3x3 | 119 | 13.9916 | 82.1920 | 17.8080 | 24.2942 |
| otsu_5x5_baseline | 119 | 13.8575 | 82.6634 | 17.3366 | 26.1332 |
| otsu_7x7 | 119 | 13.5612 | 79.1597 | 20.8403 | 29.4187 |
| otsu_no_opening | 119 | 14.0986 | 87.3627 | 12.6373 | 21.9265 |


## 3.2 Category-wise Comparison

| Category | Method | Images | Mean_Foreground_Percent | Mean_Defect_Inside_Percent | Mean_Defect_Outside_Percent |
| --- | --- | --- | --- | --- | --- |
| manipulated_front | otsu_3x3 | 24 | 13.7904 | 56.0198 | 43.9802 |
| manipulated_front | otsu_5x5_baseline | 24 | 13.7121 | 49.5801 | 50.4199 |
| manipulated_front | otsu_7x7 | 24 | 13.5732 | 44.7524 | 55.2476 |
| manipulated_front | otsu_no_opening | 24 | 13.8669 | 56.7350 | 43.2650 |
| scratch_head | otsu_3x3 | 24 | 14.1396 | 95.2024 | 4.7976 |
| scratch_head | otsu_5x5_baseline | 24 | 13.9267 | 98.0952 | 1.9048 |
| scratch_head | otsu_7x7 | 24 | 13.7098 | 96.3409 | 3.6591 |
| scratch_head | otsu_no_opening | 24 | 14.2466 | 100.0000 | 0.0000 |
| scratch_neck | otsu_3x3 | 25 | 13.8452 | 77.1442 | 22.8558 |
| scratch_neck | otsu_5x5_baseline | 25 | 13.6892 | 81.1594 | 18.8406 |
| scratch_neck | otsu_7x7 | 25 | 13.1948 | 78.8143 | 21.1857 |
| scratch_neck | otsu_no_opening | 25 | 13.9792 | 89.9954 | 10.0046 |
| thread_side | otsu_3x3 | 23 | 13.8189 | 93.5475 | 6.4525 |
| thread_side | otsu_5x5_baseline | 23 | 13.7403 | 93.2722 | 6.7278 |
| thread_side | otsu_7x7 | 23 | 13.4564 | 89.9981 | 10.0019 |
| thread_side | otsu_no_opening | 23 | 13.9142 | 93.9654 | 6.0346 |
| thread_top | otsu_3x3 | 23 | 14.3791 | 90.0572 | 9.9428 |
| thread_top | otsu_5x5_baseline | 23 | 14.2371 | 92.1082 | 7.8918 |
| thread_top | otsu_7x7 | 23 | 13.8966 | 86.6715 | 13.3285 |
| thread_top | otsu_no_opening | 23 | 14.4998 | 96.6709 | 3.3291 |


## 4. Numerical Foreground Diagnostics


## 4.1 Overall Diagnostics

| Method | Foreground_Percent | Defect_Inside_Percent | Defect_Outside_Percent | Foreground_Border_Percent | Largest_Component_Share_Percent | Connected_Component_Count | Bounding_Box_Fill_Percent | Foreground_Mean_Intensity | Outside_Mean_Intensity | Foreground_Intensity_Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| otsu_baseline | 13.8575 | 82.6634 | 17.3366 | 0.0328 | 99.9996 | 1.0084 | 37.7475 | 111.7969 | 198.6490 | 27.2094 |
| otsu_no_opening | 14.0986 | 87.3627 | 12.6373 | 0.0368 | 99.9976 | 1.0168 | 37.1796 | 112.2090 | 198.8252 | 27.3394 |


## 4.2 Category-wise Diagnostics

| Category | Method | Foreground_Percent | Defect_Inside_Percent | Defect_Outside_Percent | Foreground_Border_Percent | Largest_Component_Share_Percent | Connected_Component_Count | Bounding_Box_Fill_Percent | Foreground_Mean_Intensity | Outside_Mean_Intensity | Foreground_Intensity_Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| manipulated_front | otsu_baseline | 13.7121 | 49.5801 | 50.4199 | 0.0000 | 100.0000 | 1.0000 | 39.8491 | 105.8814 | 195.6596 | 27.1167 |
| manipulated_front | otsu_no_opening | 13.8669 | 56.7350 | 43.2650 | 0.0000 | 99.9900 | 1.0417 | 39.4773 | 106.2089 | 195.7691 | 27.2409 |
| scratch_head | otsu_baseline | 13.9267 | 98.0952 | 1.9048 | 0.0350 | 100.0000 | 1.0000 | 38.7117 | 113.5607 | 201.0763 | 27.4899 |
| scratch_head | otsu_no_opening | 14.2466 | 100.0000 | 0.0000 | 0.0446 | 100.0000 | 1.0000 | 37.9568 | 114.0816 | 201.3218 | 27.6377 |
| scratch_neck | otsu_baseline | 13.6892 | 81.1594 | 18.8406 | 0.0409 | 99.9983 | 1.0400 | 35.2160 | 114.0081 | 199.4490 | 26.8835 |
| scratch_neck | otsu_no_opening | 13.9792 | 89.9954 | 10.0046 | 0.0451 | 100.0000 | 1.0000 | 34.9714 | 114.4644 | 199.6609 | 27.0388 |
| thread_side | otsu_baseline | 13.7403 | 93.2722 | 6.7278 | 0.0435 | 100.0000 | 1.0000 | 36.9549 | 110.7584 | 198.5817 | 27.2154 |
| thread_side | otsu_no_opening | 13.9142 | 93.9654 | 6.0346 | 0.0482 | 100.0000 | 1.0000 | 36.1011 | 111.0616 | 198.7113 | 27.2949 |
| thread_top | otsu_baseline | 14.2371 | 92.1082 | 7.8918 | 0.0454 | 100.0000 | 1.0000 | 38.0927 | 114.7639 | 198.4331 | 27.3615 |
| thread_top | otsu_no_opening | 14.4998 | 96.6709 | 3.3291 | 0.0466 | 99.9980 | 1.0435 | 37.4497 | 115.2117 | 198.6147 | 27.5022 |


## 5. Manipulated-front Intensity Analysis

The intensity analysis compared defect and non-defect pixel brightness
against each image's Otsu threshold.


## 5.1 Average Intensity Findings

| Otsu_Threshold | Defect_Mean_Intensity | Normal_Mean_Intensity | Defect_Darker_Percent | Defect_Brighter_Percent | Normal_Darker_Percent | Normal_Brighter_Percent |
| --- | --- | --- | --- | --- | --- | --- |
| 150.2917 | 129.9810 | 183.5736 | 56.4562 | 43.5438 | 13.3843 | 86.6157 |


## 6. Local Contrast Inside Predicted Foreground


## 6.1 Inside-foreground Summary

| Images_Analyzed | Foreground_Percent | Defect_Retention_Percent | Defect_Pixels_Inside | Normal_Pixels_Inside | Defect_Mean_Intensity | Normal_Mean_Intensity | Defect_Median_Intensity | Normal_Median_Intensity | Defect_Mean_Local_Contrast | Normal_Mean_Local_Contrast | Defect_Median_Local_Contrast | Normal_Median_Local_Contrast | Local_Contrast_Mean_Difference | Local_Contrast_Mean_Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24.0000 | 13.8669 | 56.7350 | 33.1667 | 2238.7917 | 98.3652 | 106.4327 | 96.1250 | 103.6667 | 38.5665 | 24.8048 | 38.8333 | 24.9650 | 13.7617 | 1.5583 |


## 7. Pixel-level Defect Detection Evaluation


## 7.1 Overall Pixel-level Results

| Images_Analyzed | Total_Foreground_Pixels | Total_Defect_Pixels_Inside | Macro_Contrast_ROC_AUC | Pooled_Contrast_ROC_AUC | Macro_Intensity_ROC_AUC | Pooled_Intensity_ROC_AUC | Macro_Top10_Contrast_Precision | Macro_Top10_Contrast_Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24.0000 | 54527.0000 | 796.0000 | 0.9348 | 0.8855 | 0.4095 | 0.3575 | 0.1046 | 0.8435 |


## 7.2 Per-image Pixel-level Results

| Image | Foreground_Pixel_Count | Defect_Pixels_Inside | Contrast_ROC_AUC | Intensity_ROC_AUC | Top10_Contrast_Precision | Top10_Contrast_Recall |
| --- | --- | --- | --- | --- | --- | --- |
| 000.png | 2209 | 17 | 0.9860 | 0.5253 | 0.0769 | 1.0000 |
| 001.png | 2245 | 20 | 0.9923 | 0.4511 | 0.0889 | 1.0000 |
| 002.png | 2529 | 7 | 0.9594 | 0.5892 | 0.0277 | 1.0000 |
| 003.png | 2126 | 54 | 0.8746 | 0.1949 | 0.1643 | 0.6481 |
| 004.png | 2288 | 26 | 0.9762 | 0.3723 | 0.1092 | 0.9615 |
| 005.png | 2264 | 38 | 0.9920 | 0.2766 | 0.1674 | 1.0000 |
| 006.png | 2197 | 42 | 0.9400 | 0.3739 | 0.1500 | 0.7857 |
| 007.png | 2076 | 58 | 0.8920 | 0.3799 | 0.1779 | 0.6379 |
| 008.png | 2080 | 44 | 0.9064 | 0.3275 | 0.1538 | 0.7273 |
| 009.png | 2077 | 87 | 0.8545 | 0.2498 | 0.2548 | 0.6092 |
| 010.png | 2357 | 48 | 0.9088 | 0.2612 | 0.1271 | 0.6250 |
| 011.png | 2157 | 35 | 0.9243 | 0.2639 | 0.1157 | 0.7143 |
| 012.png | 2520 | 11 | 0.9782 | 0.4166 | 0.0437 | 1.0000 |
| 013.png | 2196 | 33 | 0.9831 | 0.3357 | 0.1500 | 1.0000 |
| 014.png | 2313 | 35 | 0.9838 | 0.3250 | 0.1466 | 0.9714 |
| 015.png | 2144 | 57 | 0.8245 | 0.3264 | 0.1349 | 0.5088 |
| 016.png | 2447 | 111 | 0.6877 | 0.2848 | 0.1265 | 0.2793 |
| 017.png | 2285 | 3 | 0.9784 | 0.6249 | 0.0131 | 1.0000 |
| 018.png | 2249 | 18 | 0.9296 | 0.7473 | 0.0667 | 0.8333 |
| 019.png | 2317 | 12 | 0.9791 | 0.4452 | 0.0517 | 1.0000 |
| 020.png | 2276 | 6 | 0.9794 | 0.6345 | 0.0263 | 1.0000 |
| 021.png | 2316 | 7 | 0.9472 | 0.6076 | 0.0302 | 1.0000 |
| 022.png | 2546 | 10 | 0.9992 | 0.3233 | 0.0392 | 1.0000 |
| 023.png | 2313 | 17 | 0.9593 | 0.4918 | 0.0690 | 0.9412 |


## 8. Main Experimental Findings

1. The Otsu no-opening variant achieved the highest average defect-pixel
   retention among the four compared foreground methods, at approximately
   87.36% across the 119 defective images.

2. Foreground segmentation remains category-dependent. The
   manipulated_front category achieved only approximately 56.74%
   defect-pixel retention with the no-opening method.

3. Defect pixels inside the predicted manipulated_front foreground had
   higher mean local contrast than non-defect foreground pixels:
   approximately 38.57 compared with 24.81.

4. In the manipulated_front pixel-level experiment, local contrast
   achieved a macro ROC-AUC of 0.9348 and pooled ROC-AUC of 0.8855.
   Raw intensity achieved 0.4095 and 0.3575 respectively.

5. Selecting the top 10% highest-contrast foreground pixels achieved
   macro recall of approximately 84.35%, but macro precision was only
   approximately 10.46%. This is useful for exploratory highlighting,
   not proof of reliable standalone defect segmentation.

## 9. Limitations

- The focused pixel-level experiment covers manipulated_front only.
- Foreground masks are predicted using Otsu thresholding; independent
  ground-truth screw masks were not available to validate segmentation.
- The defect masks identify defective regions, not necessarily the
  complete physical screw boundary.
- The exploratory pixel-level results use labeled test images and must
  not be presented as independent model generalization performance.
- Pixel-level localization metrics do not establish production-level
  image classification performance.
- These experiments do not validate a replacement for the existing
  Isolation Forest production model.

## 10. Final Recommendation

Retain the existing production model and its saved artifacts unchanged.

Keep Otsu no-opening and local contrast as experimental analysis
methods. Local contrast demonstrates promising defect-pixel ranking
performance in manipulated_front, but further independent validation
would be required before deployment.

## 11. Production Model Status

No production model retraining, threshold modification, or dashboard
modification was performed as part of this experiment consolidation.

---

**Project:** Smart Quality Intelligence  
**Report type:** Final Vision Experiment Summary  
**Resolution:** 128 × 128

