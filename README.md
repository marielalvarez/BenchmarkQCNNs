# Benchmark de redes neuronales convolucionales cuánticas

Este repositorio contiene un benchmark que compara tres redes neuronales convolucionales cuánticas (QCNN) publicadas en la literatura contra una QCNN genérica y contra redes convolucionales clásicas de tamaño comparable. Los modelos evaluados son los de Hur, Kim y Park (2022), Wei, Chen, Zhou y Long (2021), Gong, Pei, Zhang y Zhou (2024) y la arquitectura genérica de Cong, Choi y Lukin (2019).

Todos los modelos se entrenan con el mismo protocolo sobre tres tareas de clasificación binaria (Fashion-MNIST abrigo contra camisa, MNIST 4 contra 9 y MNIST 1 contra 0). Después se evalúan en condiciones cada vez más cercanas al hardware real, primero con simulación analítica, luego con un número finito de mediciones (shots) y por último con ruido de dispositivos NISQ. Las comparaciones usan intervalos de confianza bootstrap BCa, pruebas pareadas de permutación de signo y corrección de Holm.

## Estructura del repositorio

La carpeta src/qcnn_benchmark contiene el paquete de Python con toda la lógica reutilizable. Los notebooks y scripts solo lo configuran y lo ejecutan. Sus módulos son los siguientes.

- data: carga de MNIST y Fashion-MNIST mediante scikit-learn y muestreo estratificado con semilla fija (500 ejemplos de entrenamiento, 250 de validación y 500 de prueba por clase).
- representations: preprocesamiento de las imágenes, ya sea PCA escalada a [0, π] o codificación de amplitud sobre imágenes rellenadas a 32 por 32 píxeles.
- models: adaptadores de las cuatro QCNN y de las tres CNN clásicas análogas. Cada módulo expone una función predict_proba y su número total de parámetros.
- training: ciclo de entrenamiento común a todos los modelos (entropía cruzada binaria, Adam, recorte del gradiente y early stopping) y estrategias de inicialización.
- execution: ejecución con un número finito de shots y transpilación de los circuitos a un mapa de acoplamiento de 16 qubits.
- noise: canales de ruido (despolarizante, error de lectura y relajación térmica) y las siete condiciones de ruido del experimento E2.
- metrics: métricas de clasificación, de convergencia y de recursos.
- stats: bootstrap BCa, prueba de permutación pareada, corrección de Holm y tamaños de efecto.
- config: esquema de configuración de experimentos basado en Pydantic. Cada configuración tiene un hash canónico que identifica sus archivos de resultados.

La carpeta configs contiene un archivo YAML por experimento con los datasets, los modelos, el protocolo de entrenamiento, las semillas y el criterio estadístico. El campo description de cada archivo forma parte del hash, así que modificarlo desvincula la configuración de los resultados ya guardados.

La carpeta notebooks contiene los experimentos, pensados para ejecutarse en el siguiente orden.

1. 00_reproduce_hur, 00_reproduce_wei y 00_reproduce_gong comprueban que cada adaptador se acerca a la exactitud publicada en su artículo original.
2. E0A_seleccion_candidato compara a Hur, Wei y Gong en simulación analítica y elige el modelo que avanza a las siguientes fases.
3. E0_quantum_advantage compara al modelo elegido contra su CNN clásica análoga.
4. E0B_validez_aportacion compara al modelo elegido contra la QCNN genérica de Cong para ver si su diseño específico aporta una ventaja.
5. E1_finite_shots evalúa a los candidatos con 256, 1024 y 4096 shots.
6. E2_nisq_noise evalúa a los modelos con 1024 shots, transpilación y ruido.

La subcarpeta notebooks/dev guarda las primeras versiones exploratorias de las reproducciones de Hur y Wei, escritas antes de que existiera el paquete. En ellas se documenta cómo se identificó cada circuito a partir de los artículos. El archivo notebooks/e1_script_v2.py es una versión en script del notebook E1 para ejecutarlo como proceso en segundo plano.

La carpeta scripts contiene dos programas para ejecuciones largas. run_e1_expansion.py corre la matriz completa de E1 en segundo plano y puede retomarse si se interrumpe. backfill_predictive_metrics.py reentrena las corridas analíticas ya registradas para calcular sus matrices de confusión.

La carpeta results guarda los resultados de los experimentos ya ejecutados. Cada experimento produce archivos con el hash de su configuración en el nombre. Los archivos raw tienen una fila por corrida, los archivos provenance registran la configuración completa, la fecha de ejecución y las versiones de las librerías, y los archivos convergence, resources y predictive_backfill contienen métricas derivadas. También se incluyen los registros de salida de las ejecuciones de E1.

La carpeta tests contiene las pruebas unitarias del paquete. Verifican que cada modelo produce probabilidades válidas, que el gradiente fluye, que los canales de ruido y la transpilación funcionan, y que el hash de configuración coincide con el de los resultados guardados. La comparación contra los valores publicados se hace en los notebooks de reproducción, porque entrenar los circuitos es demasiado lento para una suite de pruebas.

## Instalación

El proyecto se desarrolló con Python 3.12. El archivo requirements.txt fija las versiones exactas del entorno con el que se obtuvieron los resultados.

```
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -e .
```

Los adaptadores de Hur, Cong y Gong reutilizan sin modificar el código publicado por Hur et al. bajo licencia Apache 2.0. Ese código no se incluye en el repositorio y debe descargarse en la carpeta external antes de usar esos modelos.

```
git clone https://github.com/takh04/QCNN external/hur_qcnn
```

La primera ejecución descarga MNIST y Fashion-MNIST desde OpenML, por lo que requiere conexión a internet.

## Uso

Las pruebas se ejecutan desde la raíz del repositorio.

```
./.venv/bin/python -m pytest tests
```

Los notebooks pueden abrirse desde la raíz o desde la carpeta notebooks, porque buscan las configuraciones y los resultados en ambas ubicaciones. Cada notebook de experimento tiene una variable FAST_SMOKE_TEST que reduce el número de semillas y de actualizaciones para comprobar que todo funciona antes de lanzar la corrida completa. Los experimentos con shots y con ruido pueden tardar días, así que para ellos conviene usar los scripts en segundo plano.

```
nohup python3 -u scripts/run_e1_expansion.py > results/e1_output.log 2>&1 &
```

## Referencias

Hur, T., Kim, L. y Park, D. K. (2022). Quantum convolutional neural network for classical data classification. Quantum Machine Intelligence, 4(1), 3.

Wei, S., Chen, Y., Zhou, Z. y Long, G. (2021). A Quantum Convolutional Neural Network on NISQ Devices. arXiv:2104.06918.

Gong, L., Pei, J., Zhang, T. y Zhou, N. (2024). Quantum convolutional neural network based on variational quantum circuits. Optics Communications, 550, 129993.

Cong, I., Choi, S. y Lukin, M. D. (2019). Quantum convolutional neural networks. Nature Physics, 15, 1273-1278.
