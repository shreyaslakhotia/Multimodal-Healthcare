"""Training-only context vocabulary with an explicit unknown slot."""
import numpy as np


class ContextEncoder:
    categorical = ('admission_type','first_careunit')

    def fit(self, frame):
        age = frame['age'].to_numpy(dtype=float)
        if len(age)==0 or not np.isfinite(age).all():
            raise ValueError('Expected finite training ages')
        self.mean_ = float(age.mean())
        self.scale_ = float(age.std()) or 1.
        self.vocabularies_ = {}
        self.feature_names_ = ['age','sex']
        for column in self.categorical:
            vocab = sorted(str(value) for value in frame[column].dropna().unique())
            self.vocabularies_[column] = vocab
            self.feature_names_ += [f'{column}__{value}' for value in vocab] + [f'{column}__other']
        return self

    def transform(self, frame):
        age = frame['age'].to_numpy(dtype=float)
        sex = frame['sex'].to_numpy(dtype=float)
        if not np.isfinite(age).all() or not np.isin(sex,[0,1]).all():
            raise ValueError('Context age/sex invalid')
        chunks = [((age-self.mean_)/self.scale_)[:,None],sex[:,None]]
        for column in self.categorical:
            vocab = self.vocabularies_[column]
            index = {value:i for i,value in enumerate(vocab)}
            encoded = np.zeros((len(frame),len(vocab)+1),dtype=np.float32)
            positions = [index.get(str(value),len(vocab)) for value in frame[column]]
            encoded[np.arange(len(frame)),positions] = 1
            chunks.append(encoded)
        return np.concatenate(chunks,axis=1).astype(np.float32)
