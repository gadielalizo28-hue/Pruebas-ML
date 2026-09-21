from pathlib import Path
from typing import Any
import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

class MotorInferenciaONNX:

    def __init__(self, modelo_id: str = "gpt2", ruta_exportacion: str = "./onnx_model") -> None:
        """
        Inicializa el motor de inferencia configurando el identificador del modelo 
        y la ruta en el disco donde se almacenará o leerá la versión compilada en ONNX.
        
        PARAMETROS QUE DEBES CAMBIAR:
        - modelo_id: Cambia "gpt2" por el nombre exacto del modelo preentrenado de Hugging Face 
        que requieras (por ejemplo, "meta-llama/Llama-3-8B" o un modelo ajustado tuyo).
        - ruta_exportacion: Modifica el directorio si prefieres almacenar los archivos 
        binarios optimizados en otra ruta del proyecto (por ejemplo, "./pesos_onnx").
        """
        self.modelo_id: str = modelo_id
        self.ruta_exportacion: Path = Path(ruta_exportacion)
        self.modelo: Any = None
        self.tokenizer: Any = None

    def cargar_o_exportar_modelo(self) -> None:
        """
        Verifica si la versión optimizada en ONNX ya existe en el almacenamiento local.
        Si no existe, descarga el modelo base y realiza la exportación matemática del grafo.
        Si ya existe, carga directamente los pesos compilados para acelerar el arranque.
        
        ADAPTACIONES NECESARIAS:
        - Si utilizas un modelo privado de Hugging Face o con restricciones de acceso, 
        deberás añadir el parámetro 'token="tu_token_de_huggingface"' dentro de las 
        funciones from_pretrained tanto del tokenizador como del modelo.
        """
        print("Iniciando carga del tokenizador...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelo_id)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not self.ruta_exportacion.exists():
            print("Exportando modelo base a formato ONNX...")
            self.modelo = ORTModelForCausalLM.from_pretrained(
                self.modelo_id, 
                export=True
            )
            self.modelo.save_pretrained(self.ruta_exportacion)
            print("Exportacion completada exitosamente.")
        else:
            print("Cargando modelo ONNX optimizado desde el disco...")
            self.modelo = ORTModelForCausalLM.from_pretrained(self.ruta_exportacion)
            print("Motor ONNX listo para produccion.")

    def generar_texto(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Toma una cadena de texto de entrada, la procesa mediante el tokenizador,
        ejecuta la inferencia optimizada sin gradientes y decodifica la salida resultante.
        
        PARAMETROS QUE PUEDES ADAPTAR:
        - max_tokens: Modifica el valor por defecto si tus requerimientos exigen respuestas 
        más largas o más cortas.
        - Parametros adicionales de generacion: Puedes incluir temperatura, top_p o do_sample 
        dentro del metodo self.modelo.generate() si necesitas controlar la creatividad 
        de las respuestas del modelo.
        """
        if not self.modelo or not self.tokenizer:
            raise RuntimeError("El motor ONNX no ha sido inicializado en memoria.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            salida_ids = self.modelo.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )

        texto_resultado: str = self.tokenizer.decode(salida_ids[0], skip_special_tokens=True)
        return texto_resultado