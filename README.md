# Fruit-And-Vegetable-Prediction-

## Problem Statement

Analyze the Fruit and Vegetable comprehensive dataset and develop a predictive model to accurately forecast to identify the object. The goal is to create a reliable tool that can predict or identify whether the object of image is fruit and vegetable. 

## Methods used

1. Data Visualization
2. Feature Engineering
3. Deep Learning
4. Data Validation
5. ANN Training (Sequential API/Functional API)
6. ANN Improvement (Sequential API/Functional API)

## Technologies
1. Python
2. Pandas
3. Scikit-Learn
4. Tensor-Flow

---

## Findings

### EDA

Many fruits and vegetables share similar characteristics, with dominant colors being red, orange, and green. Some items may have multiple colors.

### Model

**Baseline Model:**

Test Loss     : 1.5343

Test Accuracy : 0.4481

**Improved Model:**

Test Loss     : 0.2970

Test Accuracy : 0.9156

The results from the models above demonstrate a significant improvement.

`Improvement successful.`


### Advantages of the model:

* High accuracy
* Large training dataset
* Ability to predict 17 classes

### Model limitations:

*   Fails to predict items of the same type
*   Unable to predict logos/images in 2D format
*   Struggles with predicting items of similar shape and color

## Conclusion

Increase the capacity of classes so that the model can predict a wider and more varied range of items. Reduce data that share the same shape and characteristics that are already difficult to distinguish. Incorporate additional parameters to enhance prediction accuracy.