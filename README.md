# matrix fee estimator

inspired by https://bitcoindevkit.org/blog/2021/01/fee-estimation-for-light-clients-part-1/

## how does this work?
there are two parts to matrix. the logger and the model.

### the logger
the logger basically just records mempool transactions and waits until they are confirmed. it writes to a CSV in a format like so:
```
2023-06-16 11:05:52,0,27.14
```
let's break this down.

* `2023-06-16 11:05:52` is the date/time we saw the tx
* `0` is the blocks_seen_to_confirmed difference
* `27.14` is the fee_rate

by observing hundreds of thousands of these events and putting them into `training_data.csv`, we are able to build a dataset to train the model

### the model
keras + tensorflow do the heavy lifting here, but essentially what happens is we do some light preprocesing on the training data and then try to get convergence.
like the prior data entry, we can break this down:
* `2023-06-16 11:05:52` becomes `day_of_week` and `hour_of_day` and fed in as inputs
* the third input is the blocks_seen_to_confirmed parameter, but now it is used as a *blocks conf target*

we scale the data using `MinMaxScaler` to let the net understand the data more easily.

so for example, if we wanted to inference how much feerate we should pay to get in 2 blocks, we would do this:
```
# use the model to predict the output
new_data_df = pd.DataFrame({'day_of_week': [current_day], 'hour_of_day': [current_hour], 'block_diff': [target]})

# Convert DataFrame to a NumPy array
new_data_array = new_data_df.to_numpy()

# Scale the new_data using the scaler for input features
new_data_scaled = scaler.transform(new_data_array)

# use the model to predict the output
predicted_fee_rate = model.predict(new_data_scaled)
predicted_fee_rate = y_scaler.inverse_transform(predicted_fee_rate)
```
