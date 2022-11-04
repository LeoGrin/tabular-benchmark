class Args:
    def __init__(self, data: dict = None):
        if data is None:
            self.data = {}
        else:
            self.data = data

    def __getattr__(self, key):
        return self.data.get(key, None)
