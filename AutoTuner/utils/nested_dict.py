from collections import defaultdict


def nested_dict():
    return defaultdict(nested_dict)


class NestedDict(dict):
    """A dict-like object that auto-creates nested dicts and supports merge."""

    def __getitem__(self, key):
        # 类似 defaultdict，自动创建子字典
        if key not in self:
            self[key] = NestedDict()
        return super().__getitem__(key)

    def get_depth(self):
        """Return the depth of the nested dict."""
        v_depth = []
        for v in self.values():
            if isinstance(v, NestedDict):
                v_depth.append(v.get_depth())
        if len(v_depth) > 0:
            return 1 + max(v_depth)
        else:
            return 0

    def merge(self, other):
        """Merge another NestedDict into this one (structure assumed same)."""
        # assert self.get_depth() == other.get_depth(), f"NestedDict depth mismatch, {self.get_depth()} != {other.get_depth()}."

        for k, v in other.items():
            if k in self:
                if isinstance(v, (dict, NestedDict)) and isinstance(self[k], (dict, NestedDict)):
                    if not isinstance(self[k], NestedDict):
                        self[k] = NestedDict(self[k])
                    source = v if isinstance(v, NestedDict) else NestedDict(v)
                    self[k].merge(source)
                    continue
            self[k] = v
        return self

    def to_dict(self):
        """Convert NestedDict into a normal dict recursively."""
        return {
            k: (v.to_dict() if isinstance(v, NestedDict) else v)
            for k, v in self.items()
        }
