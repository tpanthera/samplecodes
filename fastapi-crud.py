"""

"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# In-memory database
temp_db = []


class Item(BaseModel):
    name: str
    description: str = None


# Create operation
@app.post("/items/", response_model=Item)
def create_item(item: Item):
    temp_db.append(item)
    return item


# Read operation (get all items)
@app.get("/items/", response_model=List[Item])
def read_items(skip: int = 0, limit: int = 10):
    return temp_db[skip : skip + limit]


# Read operation (get a specific item by ID)
@app.get("/items/{item_id}", response_model=Item)
def read_item(item_id: int):
    if item_id < 0 or item_id >= len(temp_db):
        raise HTTPException(status_code=404, detail="Item not found")
    return temp_db[item_id]


# Update operation
@app.put("/items/{item_id}", response_model=Item)
def update_item(item_id: int, item: Item):
    if item_id < 0 or item_id >= len(temp_db):
        raise HTTPException(status_code=404, detail="Item not found")

    temp_db[item_id] = item
    return item


# Delete operation
@app.delete("/items/{item_id}", response_model=Item)
def delete_item(item_id: int):
    if item_id < 0 or item_id >= len(temp_db):
        raise HTTPException(status_code=404, detail="Item not found")

    deleted_item = temp_db.pop(item_id)
    return deleted_item
