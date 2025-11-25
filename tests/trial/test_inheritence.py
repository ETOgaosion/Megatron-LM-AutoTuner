class A:
    """This is the object which the other two objects will inherit from."""

    def __init__(self, a):
        print(f"A.__init__ was passed a={a}")


class B(A):
    """This is one of the parent objects."""

    def __init__(self, b, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"B.__init__ was passed b={b}")


class C(A):
    """And the other one..."""

    def __init__(self, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"C.__init__ was passed c={c}")


class D(B, C):
    """And here's the problem:"""

    def __init__(self, d, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"D.__init__ was passed d={d}")


D("d", a="a", b="b", c="c")
