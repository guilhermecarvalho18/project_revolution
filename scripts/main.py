import json

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model.predict import (
    PromptTemplate,
    PERSONA_TEMPLATE,
    CONTEXT,
    llm,
    StrOutputParser,
    PredictInputSchema,
    PredictOutputSchema,
    predict,
)

app = FastAPI()

@app.post("/v0/predict")
def predict_route(input_data: PredictInputSchema):
    prediction_result = predict(input_data, PredictOutputSchema)
    prediction_result_json = json.loads(prediction_result[7:-3])
    return JSONResponse(content=jsonable_encoder({"predict": prediction_result_json}))

@app.get("/teste")
async def root():
    return {"message": "Hello World"}