import difflib

list1 = ['个人信息变更业务', '变更业务', '个人信息', '办理']
res = difflib.get_close_matches('个人信息变更业务在哪里办理呢', list1)
print(res)
