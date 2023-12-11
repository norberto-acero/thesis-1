#%%
import pandas as pd
import requests
import json

response = requests.get('https://www.datos.gov.co/resource/gt2j-8ykr.json?$query=SELECT%20fecha_reporte_web%2C%20id_de_caso%2C%20fecha_de_notificaci_n%2C%20departamento%2C%20departamento_nom%2C%20ciudad_municipio%2C%20ciudad_municipio_nom%2C%20edad%2C%20unidad_medida%2C%20sexo%2C%20fuente_tipo_contagio%2C%20ubicacion%2C%20estado%2C%20pais_viajo_1_cod%2C%20pais_viajo_1_nom%2C%20recuperado%2C%20fecha_inicio_sintomas%2C%20fecha_muerte%2C%20fecha_diagnostico%2C%20fecha_recuperado%2C%20tipo_recuperacion%2C%20per_etn_%2C%20nom_grupo_%20ORDER%20BY%20%3Aid%20ASC')
data = response.text
parse = json.loads(data)
df = pd.DataFrame(parse)