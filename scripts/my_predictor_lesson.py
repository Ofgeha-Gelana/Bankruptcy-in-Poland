# Import libraries
import gzip
import json
import pickle

import pandas as pd



# Add wrangle function from lesson 5.4
def wrangle(filename):
    # Load ARFF file
    data, meta = loadarff(filename)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Rename columns
    df.columns = [re.sub(pattern='Attr',repl='feat_',string=col) for col in df.columns]
    df.rename(columns={'class':'bankrupt'},inplace=True)
    
    # Change dtype of the Labels columns
    df['bankrupt'] = df['bankrupt'].astype(np.int64)

    return df

# Add make_predictions function from lesson 5.3
def make_predictions(data_filepath, model_filepath):
    # Wrangle JSON file
    X_test = wrangle(data_filepath)
    # Load model
    with open(model_filepath, "rb") as f:
        model = pickle.load(f)
    # Generate predictions
    y_test_pred = model.predict(X_test)
    # Put predictions into Series with name "bankrupt", and same index as X_test
    y_test_pred = pd.Series(y_test_pred, index=X_test.index, name="bankrupt")
    return y_test_pred
