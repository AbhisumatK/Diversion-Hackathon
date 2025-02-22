# first line: 29
@memory.cache
def encode_series(s):
    if s.dtype == 'O':
        le = LabelEncoder()
        return pd.Series(le.fit_transform(s))
    return s
