import pandas as pd 
from sklearn.preprocessing import OneHotEncoder  
import sklearn
from sklearn.impute import SimpleImputer, KNNImputer 

import sys

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

class Imputer:
    """A class to impute missing values in the input features.

    Real world datasets are often subject to missing covariates.
    Imputation replaces the missing values allowing downstream experiments.
    This class allows multiple strategies to impute both categorical and
    numerical/continuous covariates.

    For categorical features, the class allows:

    - **replace**: Replace all null values with a user specificed constant.
    - **ignore**: Keep all missing values as is.
    - **mode**: Replace null values with most commonly occurring variable.

    For numerical/continuous features,
    the user can choose between the following strategies:

    - **mean**: Replace all missing values with the mean in the column.
    - **median**: Replace all missing values with the median in the column.
    - **knn**: Use a k Nearest Neighbour model to predict the missing value.
    - **missforest**: Use the MissForest model to predict the null values.

    Parameters
    ----------
    cat_feat_strat : str
        Strategy for imputing categorical features.
        One of `'replace'`, `'ignore'`, `'mode'`. Default is `ignore`.
    num_feat_strat : str
        Strategy for imputing numerical/continuous features.
        One of `'mean'`, `'median'`, `'knn'`, `'missforest'`. Default is `mean`.
    remaining : str
        Strategy for handling remaining columns.
        One of `'ignore'`, `'drop'`. Default is `drop`.
    """

    _VALID_CAT_IMPUTE_STRAT = ['replace', 'ignore', 'mode']
    _VALID_NUM_IMPUTE_STRAT = ['mean', 'median', 'knn', 'missforest']
    _VALID_REMAINING_STRAT = ['ignore', 'drop']

    def __init__(self, cat_feat_strat='ignore',
                 num_feat_strat='mean', remaining='drop'):
        
        assert cat_feat_strat in Imputer._VALID_CAT_IMPUTE_STRAT
        assert num_feat_strat in Imputer._VALID_NUM_IMPUTE_STRAT
        assert remaining in Imputer._VALID_REMAINING_STRAT

        self.cat_feat_strat = cat_feat_strat
        self.num_feat_strat = num_feat_strat
        self.remaining = remaining 
        self.fitted = False

    def fit(self, data, cat_feats=None,
            num_feats=None, fill_value=-1,
            n_neighbors=5, **kwargs):
      
        if cat_feats is None: cat_feats = []
        if num_feats is None: num_feats = []

        assert len(cat_feats + num_feats) != 0, "Please specify \
        categorical and numerical features."

        self._cat_feats = cat_feats
        self._num_feats = num_feats

        df = data.copy()

        ####### REMAINING VARIABLES
        remaining_feats = set(df.columns) - set(cat_feats) - set(num_feats)

        if self.remaining == 'drop':
            df = df.drop(columns=list(remaining_feats),axis = 1)

        ####### CAT VARIABLES
        if self._cat_feats:
            if self.cat_feat_strat == 'replace':
                self._cat_base_imputer = SimpleImputer(strategy='constant',
                                                       fill_value=fill_value).fit(df[cat_feats])
            elif self.cat_feat_strat == 'mode':
                self._cat_base_imputer = SimpleImputer(strategy='most_frequent').fit(df[cat_feats])

        ####### NUM VARIABLES
        if self._num_feats:
            if self.num_feat_strat == 'mean':
                self._num_base_imputer = SimpleImputer(strategy='mean').fit(df[num_feats])
            elif self.num_feat_strat == 'median':
                self._num_base_imputer = SimpleImputer(strategy='median').fit(df[num_feats])
            elif self.num_feat_strat == 'knn':
                self._num_base_imputer = KNNImputer(n_neighbors=n_neighbors,
                                                    **kwargs).fit(df[num_feats])
            # elif self.num_feat_strat == 'missforest':
            #     from missingpy import MissForest
            #     self._num_base_imputer = MissForest(**kwargs).fit(df[num_feats])

        self.fitted = True
        return self
       
    def transform(self, data):
        assert self.fitted, "Model is not fitted yet !!!"

        df = data.copy()

        remaining_feats = set(df.columns) - set(self._cat_feats) - set(self._num_feats)
        if self.remaining == 'drop':
            df = df.drop(columns=list(remaining_feats),axis = 1)
        
        if self.cat_feat_strat != 'ignore':
            if len(self._cat_feats):
                df[self._cat_feats] = self._cat_base_imputer.transform(df[self._cat_feats])

        if len(self._num_feats):
            df[self._num_feats] = self._num_base_imputer.transform(df[self._num_feats])
        return df 

    def fit_transform(self, data, cat_feats, num_feats,
                      fill_value=-1, n_neighbors=5, **kwargs):
        """Imputes dataset using imputation strategies.

        Parameters
        ----------
        data: pandas.DataFrame
            The dataframe to be imputed.
        cat_feats: list
            List of categorical features.
        num_feats: list
            List of numerical/continuous features.
        fill_value: int
            Value to be filled if `cat_feat_strat='replace'`.
        n_neighbors: int
            Number of neighbors to be used if `num_feat_strat='knn'`.
        **kwargs
            Passed on.

        Returns:
            pandas.DataFrame: Imputed dataset.
        """

        return self.fit(data, cat_feats=cat_feats,
                        num_feats=num_feats, fill_value=fill_value).transform(data)

class Preprocessor:
    """ A composite transform involving both scaling and preprocessing.

    Parameters
    ----------
    cat_feat_strat: str
        Strategy for imputing categorical features.
    num_feat_strat: str
        Strategy for imputing numerical/continuous features.
    scaling_strategy: str
        Strategy to use for scaling numerical/continuous data. 
    remaining: str
        Strategy for handling remaining columns.
      """

    def __init__(self, cat_feat_strat='ignore', num_feat_strat='mean', remaining='drop'):
 
        self.imputer = Imputer(cat_feat_strat=cat_feat_strat,
                               num_feat_strat=num_feat_strat,
                               remaining=remaining)
          
    def fit(self, data, cat_feats, num_feats,
            fill_value=-1, n_neighbors=5, **kwargs):
        """Fit imputer and scaler to dataset."""

        self._cat_feats = cat_feats
        self._num_feats = num_feats

        self.imputer.fit(data,cat_feats=self._cat_feats,
                         num_feats=self._num_feats,
                         fill_value=fill_value,
                         n_neighbors=n_neighbors,
                         **kwargs)

        data_imputed = self.imputer.transform(data)  
        self.fitted = True
        return self

    def transform(self, data):
        """Impute and scale the dataset."""

        data_imputed = self.imputer.transform(data) 
        return data_imputed

    def fit_transform(self, data, cat_feats, num_feats,
                      fill_value=-1, n_neighbors=5, **kwargs):
        """Imputes and scales dataset.

        Parameters
        ----------
        data: pandas.DataFrame
            The dataframe to be imputed.
        cat_feats: list
            List of categorical features.
        num_feats: list
            List of numerical/continuous features. 
        fill_value: int
            Value to be filled if `cat_feat_strat='replace'`.
        n_neighbors: int
            Number of neighbors to be used if `num_feat_strat='knn'`.
        **kwargs
            Passed on.

        Returns:
        --------
            pandas.DataFrame: Imputed and scaled dataset.
        """

        output = self.imputer.fit_transform(data, cat_feats=cat_feats, num_feats=num_feats, 
                                            fill_value=fill_value, n_neighbors=n_neighbors, **kwargs) 

        return output