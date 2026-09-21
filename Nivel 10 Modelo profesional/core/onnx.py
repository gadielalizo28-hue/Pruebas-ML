from pathlib import Path #para manejar archivos como objetos en lugar de simple cadenas de texto
from typing import Dict, Any #herramientas de tipado estatico
import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM #permite cargar el lenguaje directamente sobre ONNX runtime 

class MotorInferenciaONNX:
    """
    Motor de inferencia de grado de producción optimizado con ONNX Runtime
    para máxima velocidad y menor uso de VRAM/RAM en despliegues.
    """
    def __init__(self, modelo_id: str = "gpt2", ruta_exportacion: str = "./onnx_model"): #asumira el modelo y creara una carpeta llamada onnx model para guardar el archivo optimizado
        self.modelo_id: str = modelo_id
        self.ruta_exportacion: Path = Path(ruta_exportacion) #convierte el texto de la ruta en un objeto path para iterar sobre el
        self.modelo: Any = None
        self.tokenizer: Any = None

    def cargar_o_exportar_modelo(self) -> None:
        """
        Verifica si el modelo optimizado ya existe en disco; si no, 
        lo descarga y lo exporta a formato ONNX automáticamente.
        """
        print(f"[INFO] Inicializando tokenizador para {self.modelo_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelo_id) #descarga o lee de cache el tokenizor oficial 
        
        # Asegurar token de relleno para evitar advertencias de generación
        if self.tokenizer.pad_token is None: #muchos modelos no tienen token nativo para rellenar espacios en blanco
            self.tokenizer.pad_token = self.tokenizer.eos_token 
            #si no existen igualamos el token relleno (pad_token) al token de fin de texto (eos_token), esto evita errores

        if not self.ruta_exportacion.exists(): #automatizacion: si la carpeta con el modelo optimizado no existe se corre lo de abajo
            print(f"[INFO] Exportando modelo base a ONNX en {self.ruta_exportacion}...")
            # Optimum de Hugging Face convierte el grafo de PyTorch a ONNX de forma nativa
            self.modelo = ORTModelForCausalLM.from_pretrained(
                self.modelo_id, 
                export=True #descarga el modelo y compila y reestructura el grafo matematico interno transformandolo al formato onnx
            )
            self.modelo.save_pretrained(self.ruta_exportacion) #guarda ese nuevo modelo en el disco
            print("[EXITO] Modelo exportado y guardado en formato ONNX.")
        
        else: #si el modelo existe va directo al disco local, carga el modelo onnx y el motor queda listo
            print(f"[INFO] Cargando modelo ONNX optimizado desde disco: {self.ruta_exportacion}...")
            self.modelo = ORTModelForCausalLM.from_pretrained(self.ruta_exportacion)
            print("[EXITO] Motor ONNX listo para producción.")

    def generar_texto(self, prompt: str, max_tokens: int = 50) -> str: #maximo de palabras
        """
        Ejecuta la inferencia acelerada mediante el grafo optimizado de ONNX.
        """
        if not self.modelo or not self.tokenizer: #si pasa algo anormal lanzara este error
            raise RuntimeError("[ERROR] El motor ONNX no ha sido cargado en memoria.")

        inputs = self.tokenizer(prompt, return_tensors="pt") 
        
        # Inferencia optimizada y libre de gradientes
        with torch.no_grad():
            salida_ids = self.modelo.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )

        texto_resultado: str = self.tokenizer.decode(salida_ids[0], skip_special_tokens=True)
        return texto_resultado