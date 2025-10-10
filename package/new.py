from pkg.mod1 import Yonsei


class Econ(Yonsei):
    def __init__(self, data):
        super().__init__()
        print("Economics")
        self.data = data

    def print_econ(self):
        print("AKARAKA" * self.data)
