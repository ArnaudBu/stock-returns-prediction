# Stock returns prediction through financial statements

This project use tabnet in order to try to predict, based on financial statements retrieved on Yahoo Finance, if a stock will outperform the market on a 1 year horizon.

The project uses tabnet to perform the predictions.

1. Get financial data from Yahoo Finance ([1_get_data.py](1_get_data.py))
2. Preprocess data into a clean dataset ([2_preprocess_data.py](2_preprocess_data.py))
3. Feature engineering to create a training dataset ([3_feature_eng.py](3_feature_eng.py))
4. Stock returns prediction with tabnet ([4_model.py](4_model.py))
5. Perform a backtest analysis with the model ([5_backtest.py](5_backtest.py))
6. Build and analyze different strategies based on the backtested scenarios ([6_strategies.py](6_strategies.py))