from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as rt

# Cargar los modelos ONNX
models = {
    "cacao": "rf_model_cacao.onnx",
    "cafe": "rf_model_cafe.onnx",
    "maiz": "rf_model_maiz.onnx",
}
sessions = {}

# Intentar cargar cada modelo en sesiones
for crop, model_path in models.items():
    try:
        sessions[crop] = rt.InferenceSession(model_path)
        print(f"Modelo de {crop} cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo de {crop}: {e}")

# Crear la instancia de FastAPI
app = FastAPI()

# Definir el modelo de datos para la solicitud
class PredictionRequest(BaseModel):
    crop_type: str
    parameters: dict

# Función para obtener recomendaciones para cacao basado en la clasificación
def get_cacao_advice(classification):
    advice = {
        "bajo": [
            "Realiza análisis de suelo y añade materia orgánica.",
            "Monitorea plagas y usa controles biológicos.",
            "Reemplaza plantas de bajo rendimiento y poda regularmente.",
            "Asegura un riego adecuado, especialmente en temporada seca."
        ],
        "medio": [
            "Aplica fertilizantes balanceados en varias etapas.",
            "Optimiza la recolección y fermentación para mejorar la calidad.",
            "Aplica controles preventivos de enfermedades."
        ],
        "alto": [
            "Mantén las prácticas actuales de manejo.",
            "Usa sensores y automatización para optimizar el riego.",
            "Capacita al personal en técnicas avanzadas de poscosecha.",
        ]
    }
    return advice.get(classification, [])

# Función para obtener recomendaciones para café basado en la clasificación
def get_cafe_advice(classification):
    advice = {
        "bajo": [
            "Revisa la calidad del suelo y añade nutrientes esenciales.",
            "Implementa técnicas de riego más eficientes.",
            "Realiza un control más riguroso de plagas y enfermedades."
        ],
        "medio": [
            "Optimiza las prácticas de fertilización.",
            "Ajusta el riego para maximizar la absorción de agua.",
            "Monitorea constantemente las condiciones climáticas."
        ],
        "alto": [
            "Mantén el nivel de producción actual.",
            "Considera técnicas avanzadas de recolección.",
            "Capacita al personal para mejorar la eficiencia."
        ]
    }
    return advice.get(classification, [])

def get_maiz_advice(classification):
    advice = {
        "bajo": [
            "Realiza un análisis de suelo y mejora su fertilidad con materia orgánica.",
            "Optimiza el control de plagas y enfermedades en las etapas iniciales.",
            "Revisa la eficiencia del sistema de riego y ajusta según las necesidades."
        ],
        "medio": [
            "Ajusta la densidad de siembra para maximizar el rendimiento.",
            "Realiza monitoreo constante de plagas para un control preventivo.",
            "Evalúa la posibilidad de utilizar tecnología para mejorar la eficiencia."
        ],
        "alto": [
            "Considera implementar sensores para optimizar el riego y la fertilización.",
            "Capacita al equipo en técnicas de cosecha para mejorar la calidad.",
            "Evalúa el mercado para posibles mejoras en la comercialización del maíz."
        ]
    }
    return advice.get(classification, [])

# Función para calcular el quartil de filtro para café
def calcular_filter_quartile_cafe(coffee_hectare, coffee_harvested, coffee_sold_price, coffee_harvest_loss):
    suma = coffee_hectare + coffee_harvested + coffee_sold_price + coffee_harvest_loss + coffee_harvested
    if suma < 2678.75:
        return 0
    elif 2701.25 <= suma <= 4614.50:
        return 1
    elif 4628.50 <= suma <= 8002.00:
        return 2
    elif suma > 8061.00:
        return 3

# Función para calcular el cuartil para maíz basado en la suma de ciertas características
def calcular_filter_quartile_maiz(maize_hectare, maize_harvested, maize_sold_price, maize_harvest_loss):
    suma = maize_hectare + maize_harvested + maize_sold_price + maize_harvest_loss + maize_harvested
    if suma < 1381:
        return 0
    elif 1382 <= suma <= 2880:
        return 1
    elif 2881 <= suma <= 5970:
        return 2
    else:
        return 3

def calcular_filter_maiz(maize_hectare, maize_harvested, maize_sold_price, maize_harvest_loss):
    return maize_hectare + maize_harvested + maize_sold_price + maize_harvest_loss + maize_harvested



@app.post("/predict")
def predict(request: PredictionRequest):
    crop_type = request.crop_type.lower()
    input_data = request.parameters

    # Verificar si existe el modelo para el tipo de cultivo especificado
    if crop_type not in sessions:
        raise HTTPException(status_code=400, detail="Tipo de cultivo no soportado")

    # Preparar el formato de entrada según el tipo de cultivo
    try:
        # Input específico para cacao
        if crop_type == "cacao":
            if not all(key in input_data for key in ["Area_Sembrada", "Area_Cosechada", "Produccion"]):
                raise HTTPException(status_code=400, detail="Datos de entrada incompletos para cacao")
            input_values = [[
                input_data["Area_Sembrada"],
                input_data["Area_Cosechada"],
                input_data["Produccion"]
            ]]
            # Ejecutar predicción y clasificar
            input_name = sessions[crop_type].get_inputs()[0].name
            prediction = sessions[crop_type].run(None, {input_name: input_values})[0][0]
            clasificacion = "bajo" if prediction < 0.3 else "medio" if prediction < 0.7 else "alto"

            # Obtener consejos para cacao basado en la clasificación
            consejos = get_cacao_advice(clasificacion)

            return {
                "Modelo": "Cacao",
                "Rendimiento_Predicho": float(prediction),
                "Clasificacion": clasificacion,
                "Consejos": consejos
            }

        # Input específico para café con filter_quartile y clasificación
        elif crop_type == "cafe":
            required_keys = [
                "Year", "coffee_hectare", "coffee_improved_hectare", "coffee_improved_cost",
                "coffee_hectare_fertilizer", "coffee_fertilizer_cost", "coffee_chemical_hectare",
                "coffee_chemical_cost", "coffee_machinery_hectare", "coffee_machinery_cost",
                "coffee_harvested", "coffee_sold_price", "coffee_harvest_loss"
            ]
            
            # Verificar que todos los parámetros necesarios estén presentes
            missing_keys = [key for key in required_keys if key not in input_data]
            if missing_keys:
                raise HTTPException(status_code=400, detail=f"Datos de entrada incompletos para cafe. Faltan los siguientes campos: {', '.join(missing_keys)}")

            # Calcular filter_quartile basado en los datos de entrada
            filter_quartile = calcular_filter_quartile_cafe(
                input_data["coffee_hectare"],
                input_data["coffee_harvested"],
                input_data["coffee_sold_price"],
                input_data["coffee_harvest_loss"]
            )

            # Crear el conjunto de valores de entrada incluyendo filter_quartile
            input_values = [[
                input_data["Year"], input_data["coffee_hectare"], input_data["coffee_improved_hectare"], 
                input_data["coffee_improved_cost"], input_data["coffee_hectare_fertilizer"], 
                input_data["coffee_fertilizer_cost"], input_data["coffee_chemical_hectare"],
                input_data["coffee_chemical_cost"], input_data["coffee_machinery_hectare"], 
                input_data["coffee_machinery_cost"], input_data["coffee_harvested"], 
                input_data["coffee_sold_price"], input_data["coffee_harvest_loss"], filter_quartile
            ]]
            
            # Ejecutar predicción con el modelo
            input_name = sessions[crop_type].get_inputs()[0].name
            prediction = sessions[crop_type].run(None, {input_name: input_values})[0][0]
            
            # Clasificar el rendimiento para café
            if prediction < 203:
                clasificacion = "bajo"
            elif 204 <= prediction <= 589:
                clasificacion = "medio"
            else:
                clasificacion = "alto"
            
            # Obtener consejos para café basado en la clasificación
            consejos = get_cafe_advice(clasificacion)

            return {
                "Modelo": "Cafe",
                "Rendimiento_Predicho": float(prediction),
                "Clasificacion": clasificacion,
                "Consejos": consejos,
            }











        # Input específico para maíz con filter_quartile y clasificación
        elif crop_type == "maiz":
            required_keys = [
                "Year", "maize_hectare", "maize_improved_hectare", "maize_improved_cost",
                "maize_hectare_fertilizer", "maize_fertilizer_cost", "maize_chemical_hectare",
                "maize_chemical_cost", "maize_machinery_hectare", "maize_machinery_cost",
                "maize_harvested", "maize_sold_price", "maize_harvest_loss"
            ]
            
            # Verificar que todos los parámetros necesarios estén presentes
            missing_keys = [key for key in required_keys if key not in input_data]
            if missing_keys:
                raise HTTPException(status_code=400, detail=f"Datos de entrada incompletos para maíz. Faltan los siguientes campos: {', '.join(missing_keys)}")

            # Calcular filter y filter_quartile basado en los datos de entrada
            filter = calcular_filter_maiz(
                input_data["maize_hectare"],
                input_data["maize_harvested"],
                input_data["maize_sold_price"],
                input_data["maize_harvest_loss"]
            )
            filter_quartile = calcular_filter_quartile_maiz(
                input_data["maize_hectare"],
                input_data["maize_harvested"],
                input_data["maize_sold_price"],
                input_data["maize_harvest_loss"]
            )

            # Crear el conjunto de valores de entrada incluyendo filter y filter_quartile
            input_values = [[
                input_data["Year"], input_data["maize_hectare"], input_data["maize_improved_hectare"], 
                input_data["maize_improved_cost"], input_data["maize_hectare_fertilizer"], 
                input_data["maize_fertilizer_cost"], input_data["maize_chemical_hectare"],
                input_data["maize_chemical_cost"], input_data["maize_machinery_hectare"], 
                input_data["maize_machinery_cost"], input_data["maize_harvested"], 
                input_data["maize_sold_price"], input_data["maize_harvest_loss"], 
                filter, filter_quartile
            ]]
            
            # Ejecutar predicción con el modelo
            input_name = sessions[crop_type].get_inputs()[0].name
            prediction = sessions[crop_type].run(None, {input_name: input_values})[0][0]
            
            # Clasificar el rendimiento para maíz
            if prediction < 451:
                clasificacion = "bajo"
            elif 451 <= prediction <= 1500:
                clasificacion = "medio"
            else:
                clasificacion = "alto"
            
            # Obtener consejos para maíz basado en la clasificación
            consejos = get_maiz_advice(clasificacion)

            return {
                "Modelo": "Maíz",
                "Rendimiento_Predicho": float(prediction),
                "Clasificacion": clasificacion,
                "Consejos": consejos,
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción de {crop_type}: {str(e)}")

# Endpoint raíz para prueba
@app.get("/")
def read_root():
    return {"message": "Modelo de predicciones de cultivos listo"}

# Ejecutar con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
