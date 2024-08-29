lines = []

print("请输入文本，输入'exit'结束：")

while True:
    line = input()

    if line.lower() == 'exit':
        break

    lines.append(line)

# 打印输入的文本
print("输入的文本：")
for line in lines:
    print(line)
