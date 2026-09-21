import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#automodelforcasuallm: descarga la arquitectura que eligas (gpt2, mistral y llama)
#autotokenzer: clase encargada de cargar el convertidor de texto a numeros especifico
#bits and bytsconfig: la cuantizacion
from peft import LoraConfig, get_peft_model, TaskType
#peft o parameter-eficcient Fine tunning: el estandar por hugging face para fine tunnig eficiente
#lora config: para configuar las propiedades matematicas
#get peft model: funcion que unifica un modelo congelado y los adaptadores entrenables de tu red
#tasktype: un enumerador de tareas 

def preparar_modelo_con_lora(nombre_modelo_base: str = "gpt2"): #definimos en str que modelo cargaremos
    """
    Carga un modelo preentrenado de Hugging Face y le aplica una capa LoRA 
    para hacer Fine-Tuning ultra eficiente en recursos.
    """
    print(f"[INFO] Cargando modelo base desde Hugging Face: {nombre_modelo_base}...")
    
    #puedes usar el siguiente bloque si la vram sufre mucho:

    configuracion_cuantizacion = BitsAndBytesConfig( #declaramos que la variable tendra dentro el BBYCONFIG
        load_in_4bit=True, #parametro que comprime los pesos originales a 4 bits
        bnb_4bit_compute_dtype=torch.float16, #aunque los modelos esten 4 bits los calculamos en 16 bits para precision 
        bnb_4bit_quant_type="nf4" 
    )

    # 1. Cargar el modelo base preentrenado de lenguaje
    modelo = AutoModelForCausalLM.from_pretrained( #descargamos el modelo con .from_pretrained
        nombre_modelo_base,
        quantization_config=configuracion_cuantizacion, #le pasa las indicaciones de 4bits
        device_map="auto"
    ) 
    #descarga arquitecturas entrenadas a escala masiva como GPT-2, llama, mistral
    
    # 2. Configurar los parámetros de LoRA (Low-Rank Adaptation)
    # LoRA inyecta matrices de descomposición de rango bajo en las capas de atención del Transformer
    configuracion_lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Tipo de tarea: Modelado de Lenguaje Causal (Generación de texto)
        r=8,                           # Rango (rank) de la matriz adaptadora. Valores típicos: 4, 8, 16. Mayor rango = más capacidad pero más memoria. Tambien puedes desminuir ese numero a 4 si ves mucho consumo
        lora_alpha=32,                 # Coeficiente de escalado alfa para ponderar el peso de la adaptación LoRA. lo general es que ese numero sea 2/4 * r
        target_modules=["c_attn"],     # Capas internas del modelo base a las que se les aplicará LoRA (específicas de GPT-2), tambien puedes usar para llama u otros ["q_proj"] [v_proj]                                                                
        lora_dropout=0.1,              # Tasa de dropout para prevenir sobreajuste en los adaptadores. Apaga el 10% de los valores de los adaptadores
        bias="none"                    # Especifica si los sesgos (biases) deben entrenarse ("none", "all", "lora_only").
    )                        
    #target_modules= o si quieres hacer fine tunning en todas las capas lineales para mayor precision
    #["v_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

    #rango y lora alpha = si el modelo no logra entender el estilo del texto subiras r a 16 o 32 por ende lora alpha
    #quedaria en 64 o 128, si la gpu se queda sin memoria, los bajaras a r=4 y lora_alpha=16

    # 3. Congelar el modelo base e inyectar los adaptadores LoRA entrenables
    modelo_peft = get_peft_model(modelo, configuracion_lora) 
    #metemos en una variable el modelo preentrenado y la config a la cual se le aplica el modelo congelado

    # 4. Imprimir estadísticas de parámetros entrenables vs congelados
    modelo_peft.print_trainable_parameters()
    #calcula el inv de la red e imprime en la consola cuantos parametros son entrenables en comparacion total
    
    return modelo_peft

if __name__ == "__main__":
    # Inicializar el pipeline de Fine-Tuning con LoRA
    modelo_optimizado = preparar_modelo_con_lora("gpt2")
    
    print(f"[EXITO] Modelo transformador adaptado con LoRA exitosamente.")
    
    # Prueba rápida de inferencia estructural con un tensor de tokens simulado
    # Tokenizamos un texto de prueba sencillo
    tokenizer = AutoTokenizer.from_pretrained("gpt2") #tokenizer de gpt2: convierte letras a sus ids numericos del voca del modelo
    tokenizer.pad_token = tokenizer.eos_token #esto evita errores de ejecucion comunes
    
    texto_prueba = "Artificial Intelligence engineering is" 
    inputs = tokenizer(texto_prueba, return_tensors="pt") 
    #procesa la frase. Le ordena devolver matrices directamente estructuradas en el formato de pytorch
    
    # Forward pass del modelo con LoRA integrado
    with torch.no_grad(): 
        outputs = modelo_optimizado(**inputs) #el **inputs desempaqueta el diccionario que genero tokenizer
        
    print(f"[INFO] Forma de los logits de salida del Transformer: {outputs.logits.shape}")