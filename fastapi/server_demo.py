import uvicorn
from fastapi import FastAPI, Form

app = FastAPI()
# 示例数据
items = []


# POST 请求
@app.post("/items")
async def create_item(item: str = Form(None), item2: str = Form(None)):
    items.append(item)
    items.append(item2)
    print(item)
    print(item2)
    return {"message": "Item created successfully: {}".format(str(item))}


if __name__ == "__main__":
    uvicorn.run(app, host="10.10.22.112", port=8010, log_level="debug")
