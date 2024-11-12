from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import onnxruntime as rt

# Load both ONNX models
try:
    cacao_session = rt.InferenceSession("modelo_cacao.onnx")
    print("Modelo de cacao cargado exitosamente.")
except Exception as e:
    print("Error al cargar el modelo de cacao ONNX:", e)

try:
    coffee_session = rt.InferenceSession("coffee_sales_model.onnx")
    print("Modelo de café cargado exitosamente.")
except Exception as e:
    print("Error al cargar el modelo de café ONNX:", e)

# Create the FastAPI instance
app = FastAPI()

# Define the data models for the requests
class CacaoPredictionRequest(BaseModel):
    Area_Sembrada: float
    Area_Cosechada: float
    Produccion: float

class CoffeePredictionRequest(BaseModel):
    coffee_acreage: float
    coffee_improved_acreage: float
    coffee_improved_cost: float
    coffee_acreage_fertilizer: float
    coffee_fertilizer_cost: float
    coffee_chemical_acreage: float
    coffee_chemical_cost: float
    coffee_machinery_acreage: float
    coffee_machinery_cost: float
    coffee_harvested: float
    coffee_sold_price: float
    coffee_harvest_loss: float

@app.post("/predict")
def predict(request: dict):
    # Determine if the request is for cacao or coffee based on the keys in the JSON
    if "Area_Sembrada" in request and "Area_Cosechada" in request and "Produccion" in request:
        # Convert the request to CacaoPredictionRequest model
        data = CacaoPredictionRequest(**request)
        input_data = [[data.Area_Sembrada, data.Area_Cosechada, data.Produccion]]

        # Run prediction with the cacao model
        try:
            input_name = cacao_session.get_inputs()[0].name
            prediction = cacao_session.run(None, {input_name: input_data})[0][0]
            return {"Modelo": "Cacao", "Rendimiento_Predicho": float(prediction)}
        except Exception as e:
            print("Error al realizar la predicción de cacao:", e)
            raise HTTPException(status_code=500, detail=f"Error en la predicción de cacao: {str(e)}")

    elif all(key in request for key in [
        'coffee_acreage', 'coffee_improved_acreage', 'coffee_improved_cost',
        'coffee_acreage_fertilizer', 'coffee_fertilizer_cost', 'coffee_chemical_acreage',
        'coffee_chemical_cost', 'coffee_machinery_acreage', 'coffee_machinery_cost',
        'coffee_harvested', 'coffee_sold_price', 'coffee_harvest_loss'
    ]):
        # Convert the request to CoffeePredictionRequest model
        data = CoffeePredictionRequest(**request)
        input_data = [[
            data.coffee_acreage, data.coffee_improved_acreage, data.coffee_improved_cost,
            data.coffee_acreage_fertilizer, data.coffee_fertilizer_cost, data.coffee_chemical_acreage,
            data.coffee_chemical_cost, data.coffee_machinery_acreage, data.coffee_machinery_cost,
            data.coffee_harvested, data.coffee_sold_price, data.coffee_harvest_loss
        ]]

        # Run prediction with the coffee model
        try:
            input_name = coffee_session.get_inputs()[0].name
            prediction = coffee_session.run(None, {input_name: input_data})[0][0]
            return {"Modelo": "Café", "Rendimiento_Predicho": float(prediction)}
        except Exception as e:
            print("Error al realizar la predicción de café:", e)
            raise HTTPException(status_code=500, detail=f"Error en la predicción de café: {str(e)}")

    else:
        # If the request does not match either model's format
        raise HTTPException(status_code=400, detail="Formato de JSON no reconocido. Verifique los datos de entrada.")

# Root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Modelo de predicciones de cacao y café listo"}

# Run with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
