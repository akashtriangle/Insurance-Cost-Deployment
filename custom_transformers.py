from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import LabelEncoder

# duplicate handling

class duplicate_handler(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self,X,y=None):
        return self
        
    def set_output(self,transform=None):
        return self
        
    def transform(self,X,y=None):
        X = X.copy()
        X.drop_duplicates(inplace=True)
        X.reset_index(drop=True,inplace=True)
        return X

# square root transformation
class sqrt_transformation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def set_output(self, transform=None):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            X[col] = np.sqrt(X[col].clip(lower=0))
        return X

class iqr_outlier_handler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ll_ = {}
        self.ul_ = {}
        self.median_ = {}

    def set_output(self, transform=None):
        return self

    
    
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.ll_[col] = q1 - 1.5 * iqr
            self.ul_[col] = q3 + 1.5 * iqr
            self.median_[col] = X[col].median()
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.ll_:
            ll = self.ll_[col]
            ul = self.ul_[col]
            median = self.median_[col]
            mask = (X[col] < ll) | (X[col] > ul)
            X[col] = X[col].astype(float)
            X.loc[mask, col] = median
        return X


# Column cleaner
class ColumnNameCleaner(BaseEstimator, TransformerMixin):
    def set_output(self, transform=None):
        return self
        
    def fit(self, X, y=None):
        self.columns_ = [col.split('__')[-1] for col in X.columns]
        return self

    def transform(self, X,y=None):
        X.columns = self.columns_
        return X

# class to clean column
class Column_Filterer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
        
    def set_output(self,transform=None):
        return self
        
    def fit(self,X,y=None):
        return self
        
    def transform(self,X,y=None):
        col_permitted = ['season', 'yr', 'holiday', 'weekday', 'workingday', 'weathersit','temp', 'hum', 'windspeed']
        input_missing_cols = [ col for col in col_permitted if col not in X.columns ]
        if input_missing_cols:
            raise ValueError
        else:
            X = X[col_permitted]
            return X


class ModifiedLabelEncoder(LabelEncoder):
    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        return super().transform(y).reshape(-1, 1)
 