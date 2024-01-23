class Model:
    def __init__(self, v):
        self.v = v
    @property #只读属性
    def vf(self):
        return self.v
m = Model("aabbccdd")
print(m.vf)
#m.v = 123#只读属性不允许被修改（不能修改值或删除）
print(m.vf)
del m.vf
print(m.vf)


