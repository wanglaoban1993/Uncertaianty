# Houseprice

### Contents

The current code consists of three parts: code, data and result.

Boston houseprice data is sourced from Harrison Jr, D., & Rubinfeld, D. L. (1978). Hedonic housing prices and the621
demand for clean air. Journal of Environmental Economics and Manage-622
ment, 5 , 81–102.623

### Description

#### How to use

1. `houseprice_network_data.ipynb` to show the data pre-processing and data visualization based on the current data.
2. `houseprice_network_train.ipynb` to train and save model. `houseprice_network_train_Kfold.ipynb `to verify the robust performance estimation.
3. `houseprice_SHAP_xai.ipynb` to show global and local explanations of prediction.
4. `houseprice_SHAP_xai_variances.ipynb` to show global and local explanations of uncertainty.
5. `houseprice_sensitivity.ipynb` to do sensitivity analyse.
6. `demo_houseprice.ipynb` to show global and local explanations of uncertainty with randomly selected data .

#### Environment requirements

- Python 3.8+
- CUDA: 11.7.1-cudnn8-devel-ubuntu20.04
- All the packages are already listed in the code, just to take attention to the version of  'numpy' (1.24.4) and 'shap' (0.43.0).
