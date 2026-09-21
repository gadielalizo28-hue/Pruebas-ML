from contextlib import asynccontextmanager 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.onnx import MotorInferenciaONNX # Importación modular profesional pero el optimizado

@asynccontextmanager #esta funcion controla el ciclo de vida (el encendido y apagado)
async def lifespan(app: FastAPI): 
    motor_ia.cargar_o_exportar_modelo()
    yield #para ahorrar memoria a la hora de recibir gente


motor_ia = MotorInferenciaONNX(modelo_id="gpt2", ruta_exportacion="./onnx_model")
app = FastAPI(title="API Generativa Nivel 10")

class SolicitudTexto(BaseModel):
    prompt: str
    max_palabras: int = 50

@app.post("/generar")
def endpoint_generar(solicitud: SolicitudTexto):
    try:
        # FastAPI solo llama a la función limpia, aislando la lógica de PyTorch
        resultado = motor_ia.generar_texto(promt_texto=solicitud.prompt, max_tokens=solicitud.max_palabras)
        return {"estado": "exito", "texto_generado": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))