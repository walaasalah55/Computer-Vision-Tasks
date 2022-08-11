# Features Descriptors & Matching :-
## 1- The Unique Features Extraction Using Harris Operator :
- **f_harris function**
  - ***Inputs*** : Image , Sensitivity factor .
  - ***Output*** : harris_response .

- **categorize_harris_response function**
  - ***Inputs*** : Image , harris_response , threshold .
  - ***Outputs*** : corner_indices , edges_indices .

## 2- Feature Descriptors Generation using scale invariant features (SIFT) :

- **detectAndCompute function**
  - ***Inputs*** : image , sigma , num_intervals , assumed_blur , image_border_width .
  - ***Output*** : SIFT keypoints and descriptors for the input image .



## 3- Image Features Matching using (SSD) & (NCC) :

- **Sum_Square_Difference function**
  - ***Inputs*** : the two descriptors array which returned from the sift algorithm .
  - ***Output*** : array of the matched features .

- **Normalized_Cross_Correlation function**
  - ***Inputs*** : the two descriptors array which returned from the sift algorithm .
  - ***Output*** : array of the matched features .

