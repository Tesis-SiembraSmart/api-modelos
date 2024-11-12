from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as rt

# Load ONNX models
models = {
    "cacao": "rf_model_cacao.onnx",
    "cafe": "coffee_sales_model.onnx",
    # Paths for frijol and maiz models would be added here as needed
}
sessions = {}

# Try loading each model into sessions
for crop, model_path in models.items():
    try:
        sessions[crop] = rt.InferenceSession(model_path)
        print(f"Modelo de {crop} cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo de {crop}: {e}")

# Create the FastAPI instance
app = FastAPI()

# Define data model
class PredictionRequest(BaseModel):
    crop_type: str
    parameters: dict

@app.post("/predict")
def predict(request: PredictionRequest):
    crop_type = request.crop_type.lower()
    input_data = request.parameters

    # Check if the model for the specified crop exists
    if crop_type not in sessions:
        raise HTTPException(status_code=400, detail="Tipo de cultivo no soportado")

    # Prepare input format based on crop type
    try:
        # Specific input for cacao
        if crop_type == "cacao":
            if not all(key in input_data for key in ["Area_Sembrada", "Area_Cosechada", "Produccion"]):
                raise HTTPException(status_code=400, detail="Datos de entrada incompletos para cacao")
            input_values = [[
                input_data["Area_Sembrada"],
                input_data["Area_Cosechada"],
                input_data["Produccion"]
            ]]
            # Classification logic specific to cacao
            input_name = sessions[crop_type].get_inputs()[0].name
            prediction = sessions[crop_type].run(None, {input_name: input_values})[0][0]
            clasificacion = "bajo" if prediction < 0.3 else "medio" if prediction < 0.7 else "alto"

            return {
                "Modelo": "Cacao",
                "Rendimiento_Predicho": float(prediction),
                "Clasificacion": clasificacion
            }

        # Generic input for other crops (e.g., cafe, frijol, maiz)
        elif crop_type in ["cafe", "frijol", "maiz"]:
            required_keys = [
                "acreage", "improved_acreage", "improved_cost",
                "acreage_fertilizer", "fertilizer_cost", "chemical_acreage",
                "chemical_cost", "machinery_acreage", "machinery_cost",
                "harvested", "sold_price", "harvest_loss"
            ]
            if not all(key in input_data for key in required_keys):
                raise HTTPException(status_code=400, detail=f"Datos de entrada incompletos para {crop_type}")

            input_values = [[
                input_data.get("acreage", 0.0), input_data.get("improved_acreage", 0.0), input_data.get("improved_cost", 0.0),
                input_data.get("acreage_fertilizer", 0.0), input_data.get("fertilizer_cost", 0.0), input_data.get("chemical_acreage", 0.0),
                input_data.get("chemical_cost", 0.0), input_data.get("machinery_acreage", 0.0), input_data.get("machinery_cost", 0.0),
                input_data.get("harvested", 0.0), input_data.get("sold_price", 0.0), input_data.get("harvest_loss", 0.0)
            ]]
            # Run prediction with the model
            input_name = sessions[crop_type].get_inputs()[0].name
            prediction = sessions[crop_type].run(None, {input_name: input_values})[0][0]

            return {"Modelo": crop_type.capitalize(), "Rendimiento_Predicho": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción de {crop_type}: {str(e)}")

# Root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Modelo de predicciones de cultivos listo"}

# Run with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
