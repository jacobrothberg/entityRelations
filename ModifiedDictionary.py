class ModifiedDictionary(dict):
    """
    Dictionary modified to initialize a list to store values for a given key.
    """

    def __getitem__(self, key):
        self.setdefault(key, list())
        return dict.__getitem__(self, key)

    def append(self, key):
        self.setdefault(key, 0)
        return
