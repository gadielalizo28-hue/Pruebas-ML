import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class MotorGeneracion:
    #hacemos una funcion donde el nombre del modelo es igual gpt2
    def __init__(self, nombre_modelo="gpt2"):
        self.nombre_modelo = nombre_modelo #definimos que el objeto nombre del modelo sera igual al nombre del modelo
        self.modelo = None #declaramos que el modelo es igual a none para que no reciba ningun parametro hasta que se le llame en la funcion
        self.tokenizer = None #lo mismo que modelo
    
    #declaramos una funcion para cargar el modelo
    def cargar_modelo(self):
        print("[INFO] Cargando tokenizador y modelo optimizado...")
        #tokenizer de gpt2: convierte letras a sus ids numericos del voca del modelo
        self.tokenizer = AutoTokenizer.from_pretrained(self.nombre_modelo)
        
        # Cuantización a 4 bits para proteger la VRAM
        config_4bit = BitsAndBytesConfig( #llamamos al optimizador
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        #cargamos el modelo
        self.modelo = AutoModelForCausalLM.from_pretrained( #descargamos el modelo con from_pretained
            self.nombre_modelo,
            quantization_config=config_4bit, #llamamos a la configuracion de 4bits
            device_map="auto"
        )
        print("[EXITO] Motor de IA listo.")

    def generar_texto(self, prompt_texto: str, max_tokens: int = 50) -> str: #str indica que esta funcion devolvera un texto como resultado
        #prompt_texto recibe el texto del usuario que debe dar una cadena de texto
        #max_tokens son el maximo de palabras a recibir (50)

        # 1. Convertir texto a números
        inputs = self.tokenizer(prompt_texto, return_tensors="pt").to(self.modelo.device)
        #tokenizer transforma las palabras en ids (prompt_texto), y retorne esos numeros en tensores y esos numeros
        #los mueva a la misma memoria donde vive el modelo (gpu/vram)
        
        # 2. Generar nuevos tokens sin calcular gradientes
        with torch.no_grad():
            salida_ids = self.modelo.generate(**inputs, max_new_tokens=max_tokens)
            #**inputs le pasa los numeros procesados al metodo generate para que empieze a predecir cuales son las 
            #palabras mas logicas y luego definimos el maximo de palabras/tokens
        
        # 3. Convertir números de vuelta a texto natural (como vimos antes)
        texto_final = self.tokenizer.decode(salida_ids[0], skip_special_tokens=True)
        #.decode transforma la lista de numeros generados otra vez en palabras legibles
        #salida_ids[0] selecciona la primera respuesta generada en el lote
        #luego limpiamos el texto de etiquetas

        return texto_final