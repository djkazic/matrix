# setup instructions

## logger
this one is easy, clone the repo and run ```go run main/main.go```

once you start getting data coming in, a file called `training_data.csv` should be populated with rows.

## model
first, create a virtualenv and source it.

```
python3 -m venv ./venv
source ./venv/bin/activate
```

make a symlink to the training data.

`ln -s ../logger/training_data.csv training_data.csv`

this way you don't need to constantly copy paste the file over

then, install dependencies:

`pip3 install -r requirements.txt`

once you're ready to inference, call

`python3 main.py`
