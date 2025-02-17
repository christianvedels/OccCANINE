import numpy as np
from scipy.stats import mode

from .formatter import (
    hisco_blocky5,
    construct_general_purpose_formatter
)

class EvalEngine:
    '''
    Evaluate the performance of a model
    model: model object (instance of OccCANINE)
    ground_truth: pd.DataFrame
    predicitons: pd.DataFrame
    pred_col: str (prefix of the columns contain the predictions)
    '''

    def __init__(self, model, ground_truth, predicitons, pred_col):
        self.pred_col = pred_col
        
        # Extract cols starting with pred_col
        self.y_pred = predicitons.filter(regex=pred_col)
        self.y_true = ground_truth.filter(regex=pred_col)

        # Get things from model
        self.formatter = model.formatter
        self.block_size = model.formatter.block_size
        self.system = model.system
        self.use_within_block_sep = False

        # Format
        self.y_pred = self.format(self.y_pred)
        self.y_true = self.format(self.y_true)

    def format(self, y):
        y = y.astype(str)
        for j in y.columns:
            col_j = y[j].tolist()
            for i in range(len(col_j)):
                if len(col_j[i]) == (self.block_size-1):
                    col_j[i] = '0' + col_j[i]
            
            # Store updated column
            y.loc[:, j] = col_j

        return y

    @staticmethod
    def _acc(y_true, y_pred):
        '''
        Check accuracy of one observation
        
        '''

        # Count number of y_pred in y_true
        pred_in_true = sum([x in y_true for x in y_pred])

        # Count number of y_true in y_pred
        true_in_pred = sum([x in y_pred for x in y_true])

        # Max number of preds
        max_preds = max(len(y_true), len(y_pred))

        # average of the two above divided by max
        res = (pred_in_true + true_in_pred) / (2 * max_preds)

        return res

    def accuracy(self):

        correct_predictions = 0
        for i in range(len(self.y_true)):
            y_true_i = self.y_true.iloc[i]
            y_pred_i = self.y_pred.iloc[i]

            # To list
            y_true_i = y_true_i.tolist()
            y_pred_i = y_pred_i.tolist()

            # Remove NaN
            y_true_i = [x for x in y_true_i if str(x) != 'nan']
            y_pred_i = [x for x in y_pred_i if str(x) != 'nan']

            # Remove empty strings
            y_true_i = [x for x in y_true_i if x != " "]
            y_pred_i = [x for x in y_pred_i if x != " "]
            
            # Check if any of the true codes are in the predicted codes
            correct_predictions += self._acc(y_true_i, y_pred_i)
    
        return correct_predictions / len(self.y_true)