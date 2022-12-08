# Main objective

Segment videos into parts of 
* movement
* stationary movement

## How?

1. Determine cutoff -> classification
2. add functionality that draws big bounding box around rat when it is moving (using cutoff)
3. add functionality that draws small bounding box around moving parts when stationary (using cutoff)

## 1. Cutoff (Classification)
So cutoff is basically the whole classification. If we can determine two cutoffs between movement/stationary movement/stationary, then we are done. This can either be done data driven or maybe tried with some sort of machine-learning model.

### a) Data driven:
* datapoints = preprocessed frames
* histogram over multiple videos
* extract thresholds from histogram

**Preprocessing:**
* average frames
* convert to grayscale (one channel)
* Gaussian blur?

### b) Machine learning model:

* datapoints = preprocessed frames
* features = pixels

**Preprocessing:**
* convert to grayscale (one channel)
* flatten
* normalize
* average frames

**Feature reduction**
* SelectKBest using f_classif as the argument score_func (the default)
* PCA

If we can train a model on a lot of datapoints (frames), we might be able to use the model directly to classify new frames and update the model, or to come up with some sort of decision boundary (isn't this implicit in the model though?).