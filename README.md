# Etherium_price_prediction

![panel](https://user-images.githubusercontent.com/46052843/182433980-77ea8378-2881-4303-91c0-977d8d52a439.jpg)

In this project, we try to predict the price of Ethereum for the next day. This project is part of a larger project that I have been working on in collaboration with one of my friends. In the main project, we are trying to apply more complicated methods and fine-tune the model to predict the price of Ethereum.
 
To predict the price of Etherium we decided to adopt deep learning methods. The reason for this choice is that classical time series forecasting methods proved inadequate to achieve good results on time series derived from economic data. Another reason is that deep learning methods adequately address the problem of non-stationarity for time series data. Following the research done in the paper [1], there are three main groups of deep learning architectures used in time series forecasting problems: RNN based with their different variants (LSTM, GRU), transformer-based, and convolution-based. The research then presents a comparative study between certain architectures and their performance on several datasets. From this research, the best performances were obtained by the models based on convolution-based architectures

# For this project
For the project we used the TNC (Temporal Convolution Network) model presented in the research [2]. The implementation of this model exists here https://github.com/philipperemy/keras-tcn

In the main project, we fine-tune the lag (look-back window of the model) and fine-tune the hyperparameters of the model. In the current project we train the model with the optimal lag we found without further hyperparametric tuning. For more information about the main project contact me personally (thelareular@gmail.com). 

The model was trained with batch size 1 for 100 epochs and lag 74 days. The horizon was set to 1 day (next day prediction). For the loss function, the Log-Cosh function was used.

# Data

The following variables were considered to predict the closing price of Etherium:

Open, High, Low, Close (target variable), Bitcoin price, Fear and Greed Index for Bitcoin

The price of Bitcoin was used because it has a high positive correlation with the price of Etherium. Accordingly, we thought it appropriate to take into account the price of the Fear and Greed Index. The Fear and Greed indicator reflects the general market sentiment about the fate of Bitcoin (Fear Or Greed). More about the Fear and Greed indicator can be found here https://alternative.me/crypto/fear-and-greed-index/.

Min-max normalization was performed on the original data they before were used for training and evaluation. Min-max normalization was used as it is the most commonly used method for data normalization in research dealing with cryptocurrency prediction

# Results
Four different metrics were used to evaluate the model: 
MSE, RMSE, MAE, MAPE

Below the matrixs presents the metric results:

| Metric       | Value        | 
| ------------- |:-------------:|
| MSE   | 0.001 |
| RMSE   |  0.027|
| MAE | 0.024 |
|MAPE | 7.635 |

The metric results are derived from the Test set. The results are very impressive!, considering that MAPE below 10 is widely accepted as a good performance.

The Graph below presents the train and test loss across the epochs:
![Figure_2](https://user-images.githubusercontent.com/46052843/182432517-f0572749-a393-4fbd-8619-0ae2978b494e.png)

And finally, the graph below presents the actual price compared with the predicted price, for the Test set:
![Figure_1](https://user-images.githubusercontent.com/46052843/182432965-549fcaef-ab93-41f9-b994-5b072dd18996.png)

# References
[1]: Liu, M., Zeng, A., Xu, Z., Lai, Q., & Xu, Q. (2021). Time series is a special sequence: Forecasting with sample convolution and interaction. arXiv preprint arXiv:2106.09305.

[2]: Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.


