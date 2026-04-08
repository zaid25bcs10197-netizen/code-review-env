import json
from fastapi import FastAPI
import uvicorn
from env.environment import CodeReviewEnv


def load_dataset():
    with open("dataset.json") as f:
        return json.load(f)


def main():
    dataset = load_dataset()
    env = CodeReviewEnv(dataset)
    
    app = FastAPI()
    
    @app.post("/reset")
    def reset():
        return env.reset()
    
    @app.post("/step")
    def step(action: dict):
        return env.step(action)
    
    @app.get("/state")
    def state():
        return env.state()
    
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
