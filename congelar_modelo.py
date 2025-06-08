import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Ruta del modelo exportado por Teachable Machine
saved_model_dir = "modelo_teachable/model.savedmodel"

# Carga el modelo
model = tf.saved_model.load(saved_model_dir)
infer = model.signatures["serving_default"]

# Congela (convierte a grafo con pesos fijos)
frozen_func = convert_variables_to_constants_v2(infer)
graph_def = frozen_func.graph.as_graph_def()

# Guarda como .pb congelado
with tf.io.gfile.GFile("frozen_model.pb", "wb") as f:
    f.write(graph_def.SerializeToString())

print("✅ Modelo congelado y guardado como frozen_model.pb")
