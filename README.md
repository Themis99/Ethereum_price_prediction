# Etherium_price_prediction
In this project, we try to predict the price of Ethereum for the next day. This project is part of a larger project that I have been working on in collaboration with one of my friends. In the main project, we are trying to apply more complicated methods and fine-tune the model to predict the price of Ethereum.
 
To predict the price of Etherium we decided to adopt deep learning methods. The reason for this choice is that classical time series forecasting methods proved inadequate to achieve good results on time series derived from economic data. Another reason is that deep learning methods adequately address the problem of non-stationarity for time series data. Following the research done in the paper [], there are three main groups of deep learning architectures used in time series forecasting problems: RNN based with their different variants (LSTM, GRU), transformer-based, and convolution-based. The research then presents a comparative study between certain architectures and their performance on several datasets. From this research, the best performances were obtained by the models based with convolution-based architectures

# For this project

In the main project, we fine-tune the lag (look-back window of the model) and fine-tune the hyperparameters of the model. In the current project we train the model with the optimal lag we found without further hyperparametric tuning. For more information about the main project contact me personally. 

The model was trained with batch size 1 for 100 epochs and lag 74 days. The horizon was set to 1 day (next day prediction). For the loss function the Log-Cosh function was used.
