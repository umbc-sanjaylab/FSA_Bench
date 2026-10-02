import numpy as np
from lifelines import KaplanMeierFitter
import torch, warnings    
from torch.utils.data import DataLoader, TensorDataset


##################################### Fit the training target to transform validation target #######################################################

def data_transform(y, discretiser, fit_transform=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fit_transform:
            y_trans = discretiser.fit_transform(*y)
        else:
            y_trans = discretiser.transform(*y) 
    return y_trans

############################################# Prepare the dataloader #########################################################################

def prepare_nnnph_dataloaders(data, args): 
    discretiser = Discretiser( args.num_duration, scheme="km") 

    # Fit discretiser using training target & use it to transform validation target
    train_time_trans, train_event_trans = data_transform( ( data.train.time, data.train.event ), discretiser, fit_transform=True ) 
    val_time_trans, val_event_trans = data_transform( ( data.val.time, data.val.event ), discretiser, fit_transform=False )

    train_dataset = TensorDataset( torch.as_tensor(  data.train.X, dtype=torch.float32 ), torch.as_tensor(train_time_trans), torch.as_tensor(train_event_trans) ) 
    val_dataset = TensorDataset( torch.as_tensor( data.val.X, dtype=torch.float32 ), torch.as_tensor(val_time_trans), torch.as_tensor(val_event_trans) )

    train_loader = DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True) 
    val_loader = DataLoader( val_dataset, batch_size=args.batch_size, shuffle=False) 
    return  train_loader, val_loader, discretiser

############################################# Discretizer #########################################################################


class Discretiser():
    """
    Arguments:
        cuts -- number of cuts to make.
        scheme -- either equidistant or km
    """

    def __init__(self, cuts, scheme='equidistant'):
        self._cuts = cuts
        self._scheme = scheme
        self.cuts = None

    def fit(self, durations, events):
        durations = durations.astype(np.float64)
        if self._scheme == 'equidistant':
            self.cuts = np.linspace(0, durations.max(), self._cuts, dtype=np.float64)
        elif self._scheme == 'km':
            kmf = KaplanMeierFitter()
            kmf.fit(durations, events)
            last_duration = durations.max()
            percentiles = np.linspace(kmf.predict(last_duration), kmf.predict(0), self._cuts, dtype=np.float64)
            self.cuts = np.unique([kmf.percentile(p) for p in percentiles][::-1])
        else:
            raise ValueError("Invalid scheme")
        return self

    def transform(self, durations, events):
        # -1 so that idx_durations starts from 0
        # +events so that uncensored values never have t=0?
        # clipping it so that max index falls within [0,...,self.cuts - 1]
        idx_durations = np.digitize(durations, self.cuts) - 1 + events
        idx_durations = idx_durations.clip(0, len(self.cuts) - 1)
        return (idx_durations, events)

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        durations, events = self.transform(durations, events)
        return (durations, events)

    @property
    def dim_out(self):
        if self.cuts is None:
            raise ValueError("Need to call `fit` before this is accessible.")
        return len(self.cuts)