from fastapi import FastAPI
from ml.model_logic import get_results
from schemas import TitanicInput, TitanicOutput


app = FastAPI()


@app.post("/predict")
async def predict(input_json: TitanicInput) -> TitanicOutput | None:
    raw_json = input_json.model_dump()

    try:
        pred, prob, factors = get_results(raw_json)
        result_dict = {
            'pred': pred,
            'prob': prob,
            'factors': factors,
        }
        return TitanicOutput(**result_dict)
    except Exception as e:
        print(f"Failed with exception:\n{str(e)}")
